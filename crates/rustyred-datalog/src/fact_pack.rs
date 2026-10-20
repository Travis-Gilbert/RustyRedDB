use crate::{ContentHash, Error};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub type RelationName = String;
pub type Tuple = Vec<Value>;

/// Adapters supply named rows and their content identity. Engines read this
/// immutable input; neither backend discovers facts in a graph or filesystem.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FactPack {
    pub relations: BTreeMap<RelationName, Vec<Tuple>>,
    pub hash: ContentHash,
}
impl FactPack {
    pub fn new(relations: BTreeMap<RelationName, Vec<Tuple>>) -> Self {
        let hash =
            ContentHash::digest(serde_json::to_vec(&relations).expect("JSON rows serialize"));
        Self { relations, hash }
    }
    pub fn validate(&self) -> Result<(), Error> {
        if self.hash
            != ContentHash::digest(serde_json::to_vec(&self.relations).map_err(Error::from)?)
        {
            return Err(Error(
                "Fact pack content does not match its identity".into(),
            ));
        }
        if self.relations.keys().any(|name| name.trim().is_empty()) {
            return Err(Error("Relation names cannot be empty".into()));
        }
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.relations.values().map(Vec::len).sum()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
