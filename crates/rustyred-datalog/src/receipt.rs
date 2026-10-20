use crate::{ContentHash, Error, Tuple};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub type EngineId = String;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Compiled,
    Runtime,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DerivedTuple {
    pub relation: String,
    pub tuple: Tuple,
    pub fact_id: String,
    pub rule_id: String,
    pub subject_id: String,
    pub reason: String,
    pub attributes: Value,
    pub confidence: f64,
    pub writeback_policy: String,
}

/// A closed-input assumption witnessed by the canonical relational evaluator.
/// This proves absence only from the bound fact pack and completed lower
/// stratum, never from an unobserved deployment or graph scope.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AbsenceEvidence {
    pub relation: String,
    /// None denotes an independently anonymous column, not a JSON null value.
    pub pattern: Vec<Option<Value>>,
    pub closed_relation: ContentHash,
    pub input_facts: ContentHash,
    pub fact_pack: ContentHash,
    pub rule_pack: ContentHash,
    pub stratum: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Derivation {
    pub rule: String,
    pub conclusion: String,
    pub premises: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub negative_premises: Vec<AbsenceEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RunStats {
    pub input_tuples: usize,
    pub output_tuples: usize,
    pub iterations: usize,
    pub matches: usize,
    pub wall_time_ns: u128,
    pub complete: bool,
    pub warnings: Vec<String>,
    /// A refused run never looks like a successful empty derivation.
    pub refusal: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DerivationReceipt {
    pub engine: EngineId,
    pub backend: Backend,
    pub rule_pack: ContentHash,
    pub fact_pack: ContentHash,
    pub derived: Vec<DerivedTuple>,
    pub provenance: Vec<Derivation>,
    pub stats: RunStats,
}
impl DerivationReceipt {
    pub fn ensure_complete(&self) -> Result<&Self, Error> {
        if let Some(error) = &self.stats.refusal {
            return Err(Error(error.clone()));
        }
        if !self.stats.complete {
            return Err(Error("Derivation did not reach a fixed point".into()));
        }
        Ok(self)
    }
}
