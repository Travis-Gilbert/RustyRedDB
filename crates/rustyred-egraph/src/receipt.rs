use crate::ContentHash;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum StopReason {
    FixedPoint,
    IterationLimit,
    Scheduled,
}

/// Counts come from actual egglog RunReports and complete graph serialization.
/// Elapsed time is an observation of this invocation, never an estimate.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SaturationReceipt {
    pub engine: String,
    pub rule_pack: ContentHash,
    pub iterations: usize,
    pub matches: usize,
    pub matches_per_rule: BTreeMap<String, usize>,
    pub eclasses: usize,
    pub enodes: usize,
    pub tuples: usize,
    pub wall_time_ns: u128,
    pub stop_reason: StopReason,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExtractionReceipt {
    pub engine: String,
    pub rule_pack: ContentHash,
    pub cost_model: ContentHash,
    pub root: String,
    pub term: String,
    pub term_hash: ContentHash,
    pub cost: u64,
    pub wall_time_ns: u128,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub output_count: usize,
    pub saturation: SaturationReceipt,
}
