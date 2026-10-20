use crate::{
    compiled::empty_receipt, AbsenceEvidence, Backend, ContentHash, Derivation, DerivationReceipt,
    DerivedTuple, Engine, EngineId, Error, FactPack, Tuple,
};
use egraph::{Atom, CostModel, RelationalEngine, RelationalFact, RulePack, StopReason};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

/// Text rules run in the real egglog relational evaluator. Its AST gate and
/// witness tables own validation and matched-premise provenance respectively.
pub struct Runtime {
    engine: RelationalEngine,
    rule_pack: ContentHash,
    max_iterations: usize,
}
impl Runtime {
    pub fn new(rules: impl Into<String>) -> Result<Self, Error> {
        let pack = RulePack::new(rules, CostModel::default());
        let rule_pack = pack.id.clone();
        let engine = RelationalEngine::new(pack).map_err(|error| Error(error.to_string()))?;
        Ok(Self {
            engine,
            rule_pack,
            max_iterations: 10_000,
        })
    }
    pub fn with_iteration_limit(mut self, limit: usize) -> Result<Self, Error> {
        if limit == 0 {
            return Err(Error("Runtime iteration limit must be positive".into()));
        }
        self.max_iterations = limit;
        Ok(self)
    }
    fn evaluate(&self, facts: &FactPack) -> Result<DerivationReceipt, Error> {
        facts.validate()?;
        let input: BTreeMap<_, _> = facts
            .relations
            .iter()
            .map(|(name, rows)| {
                let rows = rows
                    .iter()
                    .map(|row| row.iter().map(atom).collect())
                    .collect::<Result<Vec<Vec<Atom>>, Error>>()?;
                Ok((name.clone(), rows))
            })
            .collect::<Result<_, Error>>()?;
        let evaluated = self
            .engine
            .derive(&input, self.max_iterations)
            .map_err(|error| Error(error.to_string()))?;
        let base_ids: BTreeSet<_> = facts
            .relations
            .iter()
            .flat_map(|(name, rows)| rows.iter().map(move |row| fact_id(name, row)))
            .collect();
        let program_ids: BTreeSet<_> = evaluated.program_facts.iter().map(proof_fact_id).collect();
        let mut proofs = BTreeMap::<String, Vec<Derivation>>::new();
        for proof in evaluated.provenance {
            let conclusion = proof_fact_id(&proof.output);
            proofs
                .entry(conclusion.clone())
                .or_default()
                .push(Derivation {
                    rule: proof.rule,
                    conclusion,
                    premises: proof.premises.iter().map(proof_fact_id).collect(),
                    negative_premises: proof
                        .negative_premises
                        .into_iter()
                        .map(|absent| AbsenceEvidence {
                            relation: absent.relation,
                            pattern: absent
                                .pattern
                                .iter()
                                .map(|atom| atom.as_ref().map(value))
                                .collect(),
                            closed_relation: absent.closed_relation,
                            input_facts: absent.input_facts,
                            fact_pack: facts.hash.clone(),
                            rule_pack: absent.rule_pack,
                            stratum: absent.stratum,
                        })
                        .collect(),
                });
        }
        let mut receipt = empty_receipt(self.id(), Backend::Runtime, self.rule_pack.clone(), facts);
        for (relation, rows) in evaluated.relations {
            for row in rows {
                let tuple: Tuple = row.iter().map(value).collect();
                let id = fact_id(&relation, &tuple);
                if base_ids.contains(&id) {
                    continue;
                }
                let mut derivations = proofs.remove(&id).unwrap_or_default();
                if program_ids.contains(&id) {
                    derivations.push(Derivation {
                        rule: format!("program-fact:{}", self.rule_pack),
                        conclusion: id.clone(),
                        premises: vec![],
                        negative_premises: vec![],
                    });
                }
                derivations.sort_by(compare_derivations);
                derivations.dedup();
                let first = derivations.first().ok_or_else(|| {
                    Error("Evaluator output has no matched or program-fact proof".into())
                })?;
                let subject = tuple
                    .first()
                    .map(|value| {
                        value
                            .as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| value.to_string())
                    })
                    .unwrap_or_else(|| id.clone());
                receipt.derived.push(DerivedTuple {
                    relation: relation.clone(),
                    attributes: json!({"tuple":tuple}),
                    tuple,
                    fact_id: id,
                    rule_id: first.rule.clone(),
                    subject_id: subject,
                    reason: format!("Relational consequence of {}", first.rule),
                    confidence: 1.0,
                    writeback_policy: "read-only".into(),
                });
                receipt.provenance.extend(derivations);
            }
        }
        receipt
            .derived
            .sort_by(|a, b| (&a.relation, &a.fact_id).cmp(&(&b.relation, &b.fact_id)));
        receipt.provenance.sort_by(|a, b| {
            a.conclusion
                .cmp(&b.conclusion)
                .then_with(|| compare_derivations(a, b))
        });
        receipt.stats.output_tuples = receipt.derived.len();
        receipt.stats.iterations = evaluated.saturation.iterations;
        receipt.stats.matches = evaluated.saturation.matches;
        receipt.stats.complete = evaluated.saturation.stop_reason == StopReason::FixedPoint;
        Ok(receipt)
    }
}
impl Engine for Runtime {
    fn id(&self) -> EngineId {
        "egraph-theorem".into()
    }
    fn derive(&self, facts: &FactPack) -> DerivationReceipt {
        let start = Instant::now();
        let mut receipt = self.evaluate(facts).unwrap_or_else(|error| {
            let mut receipt =
                empty_receipt(self.id(), Backend::Runtime, self.rule_pack.clone(), facts);
            receipt.stats.refusal = Some(error.0);
            receipt
        });
        receipt.stats.wall_time_ns = start.elapsed().as_nanos();
        receipt
    }
}

fn atom(value: &Value) -> Result<Atom, Error> {
    Ok(match value {
        Value::String(value) => Atom::String(value.clone()),
        Value::Bool(value) => Atom::Bool(*value),
        Value::Number(value) if value.is_i64() => Atom::I64(value.as_i64().unwrap()),
        Value::Number(value) if value.is_f64() => Atom::F64(value.as_f64().unwrap()),
        _ => {
            return Err(Error(
                "Runtime tuples require String, i64, finite f64 or bool scalars".into(),
            ))
        }
    })
}
fn value(atom: &Atom) -> Value {
    match atom {
        Atom::String(value) => json!(value),
        Atom::I64(value) => json!(value),
        Atom::F64(value) => json!(value),
        Atom::Bool(value) => json!(value),
    }
}
fn fact_id(relation: &str, tuple: &Tuple) -> String {
    ContentHash::digest(serde_json::to_vec(&(relation, tuple)).expect("JSON fact serializes"))
        .to_string()
}
fn proof_fact_id(fact: &RelationalFact) -> String {
    fact_id(&fact.relation, &fact.tuple.iter().map(value).collect())
}

// Preserve the existing positive ordering even for escaped rule names; only
// actual negative evidence breaks a previously identical positive proof tie.
fn compare_derivations(a: &Derivation, b: &Derivation) -> std::cmp::Ordering {
    (&a.rule, &a.premises)
        .cmp(&(&b.rule, &b.premises))
        .then_with(|| {
            serde_json::to_vec(&a.negative_premises)
                .expect("typed absence serializes")
                .cmp(&serde_json::to_vec(&b.negative_premises).expect("typed absence serializes"))
        })
}
