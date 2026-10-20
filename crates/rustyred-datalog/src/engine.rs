use crate::{DerivationReceipt, EngineId, FactPack};

/// One invocation contract for fixed and user-supplied relational programs.
pub trait Engine {
    fn derive(&self, facts: &FactPack) -> DerivationReceipt;
    fn id(&self) -> EngineId;
}
