//! Theorem's pure symbolic engine. Graph storage and product policy belong to callers.
//!
//! Equality saturation and scalar relational evaluation share the pinned egglog
//! evaluator. Relational programs pass a parsed-AST admission gate before execution.
#![doc = include_str!("../README.md")]
pub mod context_pack;
mod engine;
mod receipt;
mod relational;
mod rule_pack;
mod term;

pub use engine::{Engine, Schedule, TermId};
pub use receipt::{ExecutionReceipt, ExtractionReceipt, SaturationReceipt, StopReason};
pub use relational::{
    RelationalAbsence, RelationalDerivation, RelationalEngine, RelationalFact, RelationalReceipt,
};
pub use rule_pack::{ContentHash, CostModel, RulePack};
pub use term::{Atom, FunctionRow, FunctionRows, Term, TermValue};

pub const ENGINE_ID: &str = "egraph-theorem";
pub const EGGLOG_VERSION: &str = "2.0.0";

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("egglog: {0}")]
    Egglog(String),
    #[error("invalid symbolic input: {0}")]
    Invalid(String),
    #[error("relational program refused: {0}")]
    Relational(String),
}

impl From<egglog::Error> for Error {
    fn from(value: egglog::Error) -> Self {
        Self::Egglog(value.to_string())
    }
}

#[cfg(test)]
mod tests;
