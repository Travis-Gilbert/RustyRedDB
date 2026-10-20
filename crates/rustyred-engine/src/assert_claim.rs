//! Atomic, derivation-backed Claim assertion.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

// ponytail: avoids intermediate Vec allocation from split_whitespace().collect::<Vec<_>>().join()
fn collapse_ws(s: &str, sep: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for w in s.split_whitespace() {
        if !out.is_empty() {
            out.push_str(sep);
        }
        out.push_str(w);
    }
    out
}

use crate::{
    get_derivation, stable_hash, DerivationAnnotation, DerivationGraphReader, DerivationKind,
    DerivationSpec, EdgeRecord, GraphMutation, GraphMutationBatch, GraphStoreError,
    GraphStoreResult, InMemoryGraphStore, NodeRecord, RedCoreGraphStore, ASSUMPTION_LABEL,
    CONSUMES_EDGE, DERIVATION_LABEL, OPERATOR_LABEL, PRODUCES_EDGE, UNDER_EDGE, VALUE_LABEL,
    VIA_EDGE,
};

pub const CLAIM_LABEL: &str = "Claim";
pub const ASSERTION_LABEL: &str = "Assertion";
pub const ASSERTION_RECEIPT_LABEL: &str = "AssertionReceipt";
pub const ASSERT_CLAIM_OPERATOR_ID: &str = "operator:assert-claim:v1";
pub const ASSERT_CLAIM_ENVIRONMENT: &str = "assert-claim:v1";
pub const SUPPORTED_BY_EDGE: &str = "SUPPORTED_BY";
pub const RECEIPT_FOR_EDGE: &str = "RECEIPT_FOR";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AssertClaimInput {
    pub tenant_slug: String,
    pub claim: String,
    #[serde(default)]
    pub claim_type: String,
    pub source_ref: String,
    pub supporting_derivation_ref: String,
    #[serde(default)]
    pub assumption_refs: Vec<String>,
    pub stated_confidence: f64,
    #[serde(default = "default_effective_confidence")]
    pub effective_confidence: f64,
    #[serde(default = "default_assertion_tier")]
    pub tier: String,
}

fn default_effective_confidence() -> f64 {
    1.0
}

fn default_assertion_tier() -> String {
    "provisional".to_string()
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct AssertClaimReceipt {
    pub receipt_id: String,
    pub claim_id: String,
    pub assertion_id: String,
    pub admission_derivation_id: String,
    pub supporting_derivation_ref: String,
    pub source_ref: String,
    pub assumption_refs: Vec<String>,
    pub stated_confidence: f64,
    pub effective_confidence: f64,
    pub tier: String,
    pub status: String,
    pub idempotent_replay: bool,
}

/// Owned-read plus atomic-batch contract used by embedded stores and MCP
/// adapters. Implementations must publish all mutations or none of them.
pub trait AssertClaimStore: DerivationGraphReader {
    fn commit_assert_claim_batch(&mut self, batch: GraphMutationBatch) -> GraphStoreResult<()>;
}

impl AssertClaimStore for InMemoryGraphStore {
    fn commit_assert_claim_batch(&mut self, batch: GraphMutationBatch) -> GraphStoreResult<()> {
        let mut staged = self.clone();
        for mutation in batch.mutations {
            match mutation {
                GraphMutation::NodeUpsert(node) => {
                    crate::GraphWrite::upsert_node(&mut staged, node)?;
                }
                GraphMutation::EdgeUpsert(edge) => {
                    crate::GraphWrite::upsert_edge(&mut staged, edge)?;
                }
                GraphMutation::NodeDelete(_) | GraphMutation::EdgeDelete(_) => {
                    return Err(GraphStoreError::new(
                        "assert_claim_delete_unsupported",
                        "assert-claim batches cannot contain graph deletes",
                    ));
                }
            }
        }
        *self = staged;
        Ok(())
    }
}

impl AssertClaimStore for RedCoreGraphStore {
    fn commit_assert_claim_batch(&mut self, batch: GraphMutationBatch) -> GraphStoreResult<()> {
        self.commit_batch(batch).map(|_| ())
    }
}

pub fn assert_claim(
    store: &mut (impl AssertClaimStore + ?Sized),
    mut input: AssertClaimInput,
) -> GraphStoreResult<AssertClaimReceipt> {
    normalize_assert_claim_input(&mut input)?;
    let claim_id = claim_id(&input.tenant_slug, &input.claim, &input.claim_type)?;
    let assertion_id = assertion_id(&claim_id, &input)?;
    let receipt_id = format!("assertion-receipt:{}", stable_hash(&assertion_id)?);

    if let Some(node) = store.derivation_node(&receipt_id)? {
        let mut receipt: AssertClaimReceipt =
            serde_json::from_value(node.properties).map_err(|error| {
                GraphStoreError::new("invalid_assertion_receipt", error.to_string())
            })?;
        receipt.idempotent_replay = true;
        return Ok(receipt);
    }

    let supporting = get_derivation(store, &input.supporting_derivation_ref)?;
    if supporting.spec.output_refs.is_empty() {
        return Err(GraphStoreError::new(
            "invalid_supporting_derivation",
            "supporting derivation has no output values",
        ));
    }
    for assumption_ref in &input.assumption_refs {
        let assumption = store.derivation_node(assumption_ref)?.ok_or_else(|| {
            GraphStoreError::new(
                "missing_assertion_assumption",
                format!("assertion assumption {assumption_ref} does not exist"),
            )
        })?;
        if !assumption
            .labels
            .iter()
            .any(|label| label == ASSUMPTION_LABEL)
        {
            return Err(GraphStoreError::new(
                "invalid_assertion_assumption",
                format!("assertion input {assumption_ref} is not an Assumption"),
            ));
        }
        if !assumption
            .properties
            .get("active")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            return Err(GraphStoreError::new(
                "inactive_assertion_assumption",
                format!("assertion assumption {assumption_ref} is inactive"),
            ));
        }
    }

    let claim_node = NodeRecord::new(
        &claim_id,
        [CLAIM_LABEL, VALUE_LABEL],
        json!({
            "claim_id": claim_id,
            "tenant_slug": input.tenant_slug,
            "claim": input.claim,
            "claim_type": input.claim_type,
        }),
    );
    if let Some(existing) = store.derivation_node(&claim_id)? {
        if !existing.labels.iter().any(|label| label == CLAIM_LABEL)
            || existing.properties != claim_node.properties
        {
            return Err(GraphStoreError::new(
                "claim_identity_collision",
                format!("existing node {claim_id} is not the canonical Claim"),
            ));
        }
    }

    let spec = DerivationSpec {
        kind: DerivationKind::Observational,
        operator_id: ASSERT_CLAIM_OPERATOR_ID.to_string(),
        input_refs: supporting.spec.output_refs.clone(),
        output_refs: vec![claim_id.clone()],
        assumption_refs: input.assumption_refs.clone(),
        params: json!({
            "assertion_id": assertion_id,
            "source_ref": input.source_ref,
            "supporting_derivation_ref": input.supporting_derivation_ref,
            "claim_type": input.claim_type,
            "stated_confidence": input.stated_confidence,
            "effective_confidence": input.effective_confidence,
            "tier": input.tier,
        }),
        environment_version: ASSERT_CLAIM_ENVIRONMENT.to_string(),
        annotation: DerivationAnnotation {
            confidence: input.effective_confidence,
            cost: 0.0,
        },
    };
    let admission_derivation_id = spec.content_id()?;
    let mut receipt = AssertClaimReceipt {
        receipt_id: receipt_id.clone(),
        claim_id: claim_id.clone(),
        assertion_id: assertion_id.clone(),
        admission_derivation_id: admission_derivation_id.clone(),
        supporting_derivation_ref: input.supporting_derivation_ref.clone(),
        source_ref: input.source_ref.clone(),
        assumption_refs: input.assumption_refs.clone(),
        stated_confidence: input.stated_confidence,
        effective_confidence: input.effective_confidence,
        tier: input.tier.clone(),
        status: "active".to_string(),
        idempotent_replay: false,
    };

    let mut mutations = Vec::new();
    if store.derivation_node(&claim_id)?.is_none() {
        mutations.push(GraphMutation::NodeUpsert(claim_node.clone()));
    }
    match store.derivation_node(ASSERT_CLAIM_OPERATOR_ID)? {
        Some(operator) if operator.labels.iter().any(|label| label == OPERATOR_LABEL) => {}
        Some(_) => {
            return Err(GraphStoreError::new(
                "invalid_derivation_operator",
                format!("{ASSERT_CLAIM_OPERATOR_ID} is not an Operator"),
            ))
        }
        None => mutations.push(GraphMutation::NodeUpsert(NodeRecord::new(
            ASSERT_CLAIM_OPERATOR_ID,
            [OPERATOR_LABEL],
            json!({"name": "AssertClaim"}),
        ))),
    }
    mutations.push(GraphMutation::NodeUpsert(NodeRecord::new(
        &admission_derivation_id,
        [DERIVATION_LABEL, ASSERTION_LABEL],
        serde_json::to_value(&spec)
            .map_err(|error| GraphStoreError::new("invalid_assertion", error.to_string()))?,
    )));
    mutations.push(GraphMutation::NodeUpsert(NodeRecord::new(
        &receipt_id,
        [ASSERTION_RECEIPT_LABEL],
        serde_json::to_value(&receipt).map_err(|error| {
            GraphStoreError::new("invalid_assertion_receipt", error.to_string())
        })?,
    )));
    for input_ref in &spec.input_refs {
        mutations.push(GraphMutation::EdgeUpsert(assertion_edge(
            &admission_derivation_id,
            CONSUMES_EDGE,
            input_ref,
            json!({}),
        )?));
    }
    mutations.push(GraphMutation::EdgeUpsert(assertion_edge(
        &admission_derivation_id,
        PRODUCES_EDGE,
        &claim_id,
        json!({"expected_content_hash": claim_node.content_address()?}),
    )?));
    mutations.push(GraphMutation::EdgeUpsert(assertion_edge(
        &admission_derivation_id,
        VIA_EDGE,
        ASSERT_CLAIM_OPERATOR_ID,
        json!({}),
    )?));
    for assumption_ref in &spec.assumption_refs {
        mutations.push(GraphMutation::EdgeUpsert(assertion_edge(
            &admission_derivation_id,
            UNDER_EDGE,
            assumption_ref,
            json!({}),
        )?));
    }
    mutations.push(GraphMutation::EdgeUpsert(assertion_edge(
        &admission_derivation_id,
        SUPPORTED_BY_EDGE,
        &input.supporting_derivation_ref,
        json!({}),
    )?));
    mutations.push(GraphMutation::EdgeUpsert(assertion_edge(
        &receipt_id,
        RECEIPT_FOR_EDGE,
        &admission_derivation_id,
        json!({}),
    )?));

    store.commit_assert_claim_batch(GraphMutationBatch::new(mutations))?;
    receipt.idempotent_replay = false;
    Ok(receipt)
}

pub fn claim_id(tenant_slug: &str, claim: &str, claim_type: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "claim:{}",
        stable_hash(json!({
            "tenant_slug": tenant_slug,
            "claim": normalize_claim_text(claim),
            "claim_type": claim_type.trim(),
        }))?
    ))
}

pub fn is_asserted_claim(
    store: &(impl DerivationGraphReader + ?Sized),
    claim_id: &str,
) -> GraphStoreResult<bool> {
    let Some(claim) = store.derivation_node(claim_id)? else {
        return Ok(false);
    };
    if !claim.labels.iter().any(|label| label == CLAIM_LABEL) {
        return Ok(false);
    }
    for producer in store
        .derivation_neighbors(crate::NeighborQuery::in_(claim_id).with_edge_type(PRODUCES_EDGE))?
    {
        let Some(assertion) = store.derivation_node(&producer.node_id)? else {
            continue;
        };
        if !assertion
            .labels
            .iter()
            .any(|label| label == ASSERTION_LABEL)
        {
            continue;
        }
        let receipts = store.derivation_neighbors(
            crate::NeighborQuery::in_(&producer.node_id).with_edge_type(RECEIPT_FOR_EDGE),
        )?;
        if receipts.into_iter().any(|hit| {
            store
                .derivation_node(&hit.node_id)
                .ok()
                .flatten()
                .is_some_and(|node| {
                    node.labels
                        .iter()
                        .any(|label| label == ASSERTION_RECEIPT_LABEL)
                })
        }) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn assertion_id(claim_id: &str, input: &AssertClaimInput) -> GraphStoreResult<String> {
    Ok(format!(
        "assertion:{}",
        stable_hash(json!({
            "claim_id": claim_id,
            "source_ref": input.source_ref,
            "supporting_derivation_ref": input.supporting_derivation_ref,
            "assumption_refs": input.assumption_refs,
        }))?
    ))
}

fn normalize_assert_claim_input(input: &mut AssertClaimInput) -> GraphStoreResult<()> {
    input.tenant_slug = input.tenant_slug.trim().to_string();
    input.claim = normalize_claim_text(&input.claim);
    input.claim_type = input.claim_type.trim().to_string();
    input.source_ref = input.source_ref.trim().to_string();
    input.supporting_derivation_ref = input.supporting_derivation_ref.trim().to_string();
    input
        .assumption_refs
        .retain(|value| !value.trim().is_empty());
    input.assumption_refs.sort();
    input.assumption_refs.dedup();
    input.tier = input.tier.trim().to_ascii_lowercase();
    for (field, value) in [
        ("tenant_slug", input.tenant_slug.as_str()),
        ("claim", input.claim.as_str()),
        ("source_ref", input.source_ref.as_str()),
        (
            "supporting_derivation_ref",
            input.supporting_derivation_ref.as_str(),
        ),
        ("tier", input.tier.as_str()),
    ] {
        if value.is_empty() {
            return Err(GraphStoreError::new(
                "invalid_assert_claim",
                format!("{field} must not be empty"),
            ));
        }
    }
    for (field, value) in [
        ("stated_confidence", input.stated_confidence),
        ("effective_confidence", input.effective_confidence),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(GraphStoreError::new(
                "invalid_assert_claim",
                format!("{field} must be finite and between zero and one"),
            ));
        }
    }
    Ok(())
}

fn normalize_claim_text(value: &str) -> String {
    collapse_ws(&value, " ")
}

fn assertion_edge(
    from_id: &str,
    edge_type: &str,
    to_id: &str,
    properties: Value,
) -> GraphStoreResult<EdgeRecord> {
    Ok(EdgeRecord::new(
        format!(
            "derivation-edge:{}",
            stable_hash((from_id, edge_type, to_id))?
        ),
        from_id,
        edge_type,
        to_id,
        properties,
    ))
}
