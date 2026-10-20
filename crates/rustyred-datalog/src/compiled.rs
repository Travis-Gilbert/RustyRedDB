use crate::{
    legacy, Backend, ContentHash, Derivation, DerivationReceipt, DerivedTuple, Engine, EngineId,
    Error, FactPack, RunStats,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, time::Instant};

/// The fifteen current fixed rules: the ten original rules and five retained
/// civic/identity rules. The relational evaluation lives with the engine.
#[derive(Clone, Debug, Default)]
pub struct Compiled {
    rule_ids: Vec<String>,
}
impl Compiled {
    pub fn new(rule_ids: Vec<String>) -> Self {
        Self { rule_ids }
    }
    pub fn rule_ids(&self) -> Vec<String> {
        if self.rule_ids.is_empty() {
            legacy::DATALOG_RULE_IDS
                .iter()
                .map(|id| (*id).into())
                .collect()
        } else {
            self.rule_ids.clone()
        }
    }
    pub fn rule_pack(&self) -> ContentHash {
        ContentHash::digest(
            serde_json::to_vec(&(
                "compiled-datafrog/1",
                self.rule_ids(),
                include_str!("legacy.rs"),
            ))
            .expect("rule metadata serializes"),
        )
    }
    fn evaluate(&self, facts: &FactPack) -> Result<Value, Error> {
        facts.validate()?;
        let rows = legacy_rows(facts)?;
        legacy::derive_datalog_receipt(&json!({"facts":rows,"rule_ids":self.rule_ids()}))
            .map_err(Error)
    }
}
impl Engine for Compiled {
    fn id(&self) -> EngineId {
        "rust-datafrog-core".into()
    }
    fn derive(&self, facts: &FactPack) -> DerivationReceipt {
        let start = Instant::now();
        let mut receipt = empty_receipt(self.id(), Backend::Compiled, self.rule_pack(), facts);
        match self.evaluate(facts) {
            Err(error) => receipt.stats.refusal = Some(error.0),
            Ok(value) => {
                for row in value["derived_facts"]
                    .as_array()
                    .expect("fixed-rule output array")
                {
                    let string = |key| row[key].as_str().expect("fixed-rule text field").to_owned();
                    let fact_id = string("fact_id");
                    receipt.provenance.push(Derivation {
                        rule: string("rule_id"),
                        conclusion: fact_id.clone(),
                        negative_premises: vec![],
                        premises: row["dependency_fact_ids"]
                            .as_array()
                            .expect("proof inputs")
                            .iter()
                            .map(|id| id.as_str().expect("proof input ID").to_owned())
                            .collect(),
                    });
                    receipt.derived.push(DerivedTuple {
                        relation: string("relation"),
                        tuple: vec![row["subject_id"].clone(), row["attributes"].clone()],
                        fact_id,
                        rule_id: string("rule_id"),
                        subject_id: string("subject_id"),
                        reason: string("reason"),
                        attributes: row["attributes"].clone(),
                        confidence: row["confidence"].as_f64().expect("fixed-rule confidence"),
                        writeback_policy: string("writeback_policy"),
                    });
                }
                receipt.stats.warnings = value["warnings"]
                    .as_array()
                    .expect("warning array")
                    .iter()
                    .map(|warning| warning.as_str().expect("text warning").to_owned())
                    .collect();
                receipt.stats.complete = true;
                // Static Relation programs evaluate once, without a recursive
                // datafrog Iteration. Proof count measures emitted conclusions.
                receipt.stats.iterations = 1;
                receipt.stats.matches = receipt.provenance.len();
            }
        }
        receipt.stats.output_tuples = receipt.derived.len();
        receipt.stats.wall_time_ns = start.elapsed().as_nanos();
        receipt
    }
}

pub(crate) fn empty_receipt(
    engine: EngineId,
    backend: Backend,
    rule_pack: ContentHash,
    facts: &FactPack,
) -> DerivationReceipt {
    DerivationReceipt {
        engine,
        backend,
        rule_pack,
        fact_pack: facts.hash.clone(),
        derived: vec![],
        provenance: vec![],
        stats: RunStats {
            input_tuples: facts.len(),
            output_tuples: 0,
            iterations: 0,
            matches: 0,
            wall_time_ns: 0,
            complete: false,
            warnings: vec![],
            refusal: None,
        },
    }
}

/// Legacy fact adapters use [entity_id, attributes, source_ref, fact_id].
pub fn fact_pack_from_legacy_rows(raw: &[Value]) -> Result<FactPack, Error> {
    let mut relations = BTreeMap::<String, Vec<Vec<Value>>>::new();
    for row in legacy::normalize_facts(raw).map_err(Error)? {
        relations
            .entry(
                row["relation"]
                    .as_str()
                    .expect("normalized relation")
                    .into(),
            )
            .or_default()
            .push(vec![
                row["entity_id"].clone(),
                row["attributes"].clone(),
                row["source_ref"].clone(),
                row["fact_id"].clone(),
            ]);
    }
    Ok(FactPack::new(relations))
}
fn legacy_rows(facts: &FactPack) -> Result<Vec<Value>, Error> {
    let mut rows = Vec::new();
    for (relation, tuples) in &facts.relations {
        for tuple in tuples {
            if tuple.len() != 4
                || !tuple[0].is_string()
                || !tuple[1].is_object()
                || !tuple[2].is_string()
                || !tuple[3].is_string()
            {
                return Err(Error(
                    "Compiled rows require entity, attributes, source and fact identity".into(),
                ));
            }
            rows.push(
                json!({"relation":relation,"entity_id":tuple[0],"attributes":tuple[1],
                "source_ref":tuple[2],"fact_id":tuple[3]}),
            );
        }
    }
    Ok(rows)
}

/// Explicit compatibility projection for the unchanged Python byte-parity API.
/// Its historic engine field remains untouched; new consumers use derive().
pub fn derive_legacy_receipt(payload: &Value) -> Result<Value, Error> {
    let (raw, rules) = legacy::datalog_payload_parts(payload).map_err(Error)?;
    let normalized = legacy::normalize_facts(raw).map_err(Error)?;
    let facts = fact_pack_from_legacy_rows(raw)?;
    let engine = Compiled::new(rules.unwrap_or_default());
    let receipt = engine.derive(&facts);
    receipt.ensure_complete()?;
    let rows: Vec<_> = receipt.derived.iter().zip(&receipt.provenance).map(|(row, proof)| json!({
        "rule_id":row.rule_id,"relation":row.relation,"subject_id":row.subject_id,
        "reason":row.reason,"dependency_fact_ids":proof.premises,"attributes":row.attributes,
        "confidence":row.confidence,"writeback_policy":row.writeback_policy,"fact_id":row.fact_id,
    })).collect();
    Ok(json!({"engine":"python-reference-datalog",
        "fact_pack_hash":legacy::fact_pack_hash_for_facts(&normalized).map_err(Error)?,
        "rule_ids":engine.rule_ids(),"derived_count":rows.len(),"derived_facts":rows,
        "warnings":receipt.stats.warnings,"writeback_policy":"read-only"}))
}
