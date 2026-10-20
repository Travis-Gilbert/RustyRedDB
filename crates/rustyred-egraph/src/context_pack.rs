//! The context consumer's actual equality rules and compatibility projection.
//!
//! Item eligibility and ordered duplicate removal are rules in the pinned
//! evaluator. Rust supplies immutable item facts and reads extracted indices.
//! Legacy token accounting is distinct from the positive structural extraction
//! cost: the original adapter canonicalized even zero/negative-cost items.
use crate::{
    Atom, CostModel, Engine, Error, ExtractionReceipt, RulePack, SaturationReceipt, Schedule,
    StopReason, TermValue,
};
use egglog::ast::{Action, Command, Expr, RustSpan, Span};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const PROGRAM: &str = r#"
(datatype ContextList (ContextNil) (ContextCons i64 ContextList))
(relation ContextItem (i64 String bool bool))
(relation ContextPresent (i64 String))
(ruleset drop_empty_optional)
(ruleset dedupe_same_obligation)
(unstable-combined-ruleset context_pack drop_empty_optional dedupe_same_obligation)
(rule ((ContextItem i key false false) (= list (ContextCons i rest)))
      ((union list rest)) :ruleset drop_empty_optional :name "drop_empty_optional")
(rule ((ContextItem i key true text)) ((ContextPresent i key))
      :ruleset dedupe_same_obligation :name "required_item_is_present")
(rule ((ContextItem i key hard true)) ((ContextPresent i key))
      :ruleset dedupe_same_obligation :name "nonempty_item_is_present")
(rule ((ContextPresent earlier key) (ContextItem later key false true)
       (< earlier later) (= list (ContextCons later rest)))
      ((union list rest)) :ruleset dedupe_same_obligation :name "dedupe_same_obligation")
"#;

/// Canonical context rules, usable by the shared engine resolver.
pub fn rule_pack() -> RulePack {
    RulePack::new(
        PROGRAM,
        CostModel {
            constructor_costs: BTreeMap::from([
                ("ContextCons".into(), 1),
                ("ContextNil".into(), 0),
            ]),
        },
    )
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContextPackInput {
    #[serde(default = "default_expression_id")]
    pub expression_id: String,
    pub items: Vec<Value>,
    #[serde(default)]
    pub cost_config: Value,
}
fn default_expression_id() -> String {
    "native-context-pack".into()
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContextPackResult {
    /// Existing adapter fields, with stable actual engine identities in metadata.
    pub receipt: Value,
    pub saturations: Vec<SaturationReceipt>,
    pub extractions: Vec<ExtractionReceipt>,
}

pub fn extract_json(payload: &str) -> Result<String, Error> {
    let input = serde_json::from_str(payload)
        .map_err(|error| Error::Invalid(format!("context pack JSON: {error}")))?;
    let result = extract(&input)?;
    let mut receipt = result.receipt;
    receipt["native_evidence"] =
        json!({"saturations":result.saturations,"extractions":result.extractions});
    serde_json::to_string(&receipt).map_err(|error| Error::Invalid(error.to_string()))
}

pub fn extract(input: &ContextPackInput) -> Result<ContextPackResult, Error> {
    extract_with_engine(Engine::new(rule_pack())?, input)
}

/// Consume the resolver's actual fresh canonical engine. A prior command can
/// register rules without adding tuples, so a tuple-count check is insufficient.
pub fn extract_with_engine(
    mut engine: Engine,
    input: &ContextPackInput,
) -> Result<ContextPackResult, Error> {
    if engine.rule_pack() != &rule_pack() || !engine.is_pristine() {
        return Err(Error::Invalid(
            "context extraction requires the untouched canonical rule pack".into(),
        ));
    }
    let expression_id = if input.expression_id.is_empty() {
        "native-context-pack"
    } else {
        &input.expression_id
    };
    let weight = input
        .cost_config
        .get("item_weight")
        .and_then(Value::as_f64)
        .unwrap_or(0.05);
    if !weight.is_finite() {
        return Err(Error::Invalid("non-finite context item weight".into()));
    }
    let pack = rule_pack();
    let costs = pack.cost_model.clone();
    let metadata = json!({"native_backend":"rust-egg-context-pack","rule_pack":pack.id,"cost_model":costs.content_hash()});

    let mut facts = Vec::new();
    for (index, item) in input.items.iter().enumerate() {
        let index = i64::try_from(index)
            .map_err(|_| Error::Invalid("context item count exceeds i64".into()))?;
        let key = item_key(item);
        let hard = item
            .get("hard_required")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let text = !item_text(item).trim().is_empty();
        let args = [
            Atom::I64(index),
            Atom::String(key),
            Atom::Bool(hard),
            Atom::Bool(text),
        ]
        .iter()
        .map(|atom| Ok(Expr::Lit(egglog::span!(), atom.literal()?)))
        .collect::<Result<Vec<_>, Error>>()?;
        facts.push(Command::Action(Action::Expr(
            egglog::span!(),
            Expr::Call(egglog::span!(), "ContextItem".into(), args),
        )));
    }
    engine.execute_commands(facts)?;
    let mut expression = "(ContextNil)".to_owned();
    for index in (0..input.items.len()).rev() {
        expression = format!("(ContextCons {index} {expression})");
    }
    let root = engine.root(&expression)?;
    let original_cost = total_cost(&input.items, weight)?;
    let input_hash = expression_hash(expression_id, &input.items, &json!({}));
    let mut previous = input.items.clone();
    let mut trace = Vec::new();
    let mut saturations = Vec::new();
    let mut extractions = Vec::new();
    for (name, reason) in [
        (
            "drop_empty_optional",
            "Removed optional empty context items without changing represented obligations.",
        ),
        (
            "dedupe_same_obligation",
            "Removed duplicate non-required context items with the same obligation and channel.",
        ),
    ] {
        // Finite ordered input: each round can expose at most one additional
        // suffix; the extra rounds cover relation propagation and quiescence.
        let limit = input
            .items
            .len()
            .checked_add(8)
            .ok_or_else(|| Error::Invalid("context schedule overflow".into()))?;
        let saturation = engine.saturate(Schedule::FixedPoint {
            rulesets: vec![name.into()],
            max_iterations: limit,
        })?;
        if saturation.stop_reason != StopReason::FixedPoint {
            return Err(Error::Invalid(format!(
                "context {name} did not reach a fixed point"
            )));
        }
        let (term, extraction) = engine.extract(root, &costs)?;
        let current = materialize(&term.value, &input.items)?;
        if current != previous {
            let removed = previous
                .len()
                .checked_sub(current.len())
                .ok_or_else(|| Error::Invalid("context rewrite added items".into()))?;
            trace.push(json!({
                "rule_id":name,
                "before_hash":expression_hash(expression_id,&previous,&json!({})),
                "after_hash":expression_hash(expression_id,&current,&json!({})),
                "reason":reason,
                "delta_cost":round6(total_cost(&current,weight)?-total_cost(&previous,weight)?),
                "data":{"removed_count":removed},
            }));
        }
        previous = current;
        saturations.push(saturation);
        extractions.push(extraction);
    }
    let output_hash = expression_hash(expression_id, &previous, &metadata);
    let receipt = json!({
        "engine":"egraph-theorem","native_backend":"rust-egg-context-pack",
        "input_hash":input_hash,"output_hash":output_hash,"domain":"context_pack","equivalent":true,
        "original_cost":round6(original_cost),"extracted_cost":round6(total_cost(&previous,weight)?),
        "rewrite_trace":trace,
        "extraction":{"expression_id":expression_id,"domain":"context_pack","items":previous,"metadata":metadata,"expression_hash":output_hash},
    });
    Ok(ContextPackResult {
        receipt,
        saturations,
        extractions,
    })
}

fn materialize(term: &TermValue, items: &[Value]) -> Result<Vec<Value>, Error> {
    let mut result = Vec::new();
    let mut cursor = term;
    let mut prior = None;
    loop {
        match cursor {
            TermValue::App(name, children) if name == "ContextNil" && children.is_empty() => {
                return Ok(result)
            }
            TermValue::App(name, children) if name == "ContextCons" && children.len() == 2 => {
                let TermValue::Atom(Atom::I64(index)) = &children[0] else {
                    return Err(Error::Invalid("context index is not i64".into()));
                };
                let index = usize::try_from(*index)
                    .map_err(|_| Error::Invalid("negative extracted context index".into()))?;
                if prior.is_some_and(|prior| prior >= index) {
                    return Err(Error::Invalid(
                        "context extraction changed input order".into(),
                    ));
                }
                result.push(
                    items
                        .get(index)
                        .ok_or_else(|| {
                            Error::Invalid("context index outside original input".into())
                        })?
                        .clone(),
                );
                prior = Some(index);
                cursor = &children[1];
            }
            _ => return Err(Error::Invalid("unexpected context extraction term".into())),
        }
    }
}

fn field<'a>(item: &'a Value, key: &str) -> &'a str {
    item.get(key).and_then(Value::as_str).unwrap_or("")
}
fn item_text(item: &Value) -> &str {
    let text = field(item, "text");
    if text.is_empty() {
        field(item, "summary")
    } else {
        text
    }
}
fn item_key(item: &Value) -> String {
    let first = |keys: &[&str]| {
        keys.iter()
            .map(|key| field(item, key))
            .find(|value| !value.is_empty())
            .unwrap_or("")
    };
    format!(
        "{}\u{1f}{}\u{1f}{}",
        field(item, "channel"),
        first(&["obligation_id", "evidence_id", "id"]),
        first(&["semantic_hash", "content_hash", "text"])
    )
}
fn total_cost(items: &[Value], weight: f64) -> Result<f64, Error> {
    let total = items
        .iter()
        .map(|item| {
            item.get("tokens")
                .and_then(Value::as_f64)
                .unwrap_or_else(|| {
                    let text = item_text(item);
                    if text.is_empty() {
                        1.0
                    } else {
                        text.len() as f64 / 4.0
                    }
                })
                + weight
        })
        .sum::<f64>();
    if total.is_finite() {
        Ok(total)
    } else {
        Err(Error::Invalid("context cost overflow".into()))
    }
}
fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}
fn expression_hash(id: &str, items: &[Value], metadata: &Value) -> String {
    let bytes = serde_json::to_vec(
        &json!({"expression_id":id,"domain":"context_pack","items":items,"metadata":metadata}),
    )
    .expect("JSON serializes");
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests;
