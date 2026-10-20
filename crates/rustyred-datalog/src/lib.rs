#![doc = include_str!("../README.md")]
mod compiled;
mod engine;
mod fact_pack;
mod legacy;
mod receipt;
mod runtime;
pub use compiled::{derive_legacy_receipt, fact_pack_from_legacy_rows, Compiled};
pub use egraph::ContentHash;
pub use engine::Engine;
pub use fact_pack::{FactPack, RelationName, Tuple};
pub use legacy::DATALOG_RULE_IDS;
pub use receipt::{
    AbsenceEvidence, Backend, Derivation, DerivationReceipt, DerivedTuple, EngineId, RunStats,
};
pub use runtime::Runtime;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error(pub String);
impl std::fmt::Display for Error {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}
