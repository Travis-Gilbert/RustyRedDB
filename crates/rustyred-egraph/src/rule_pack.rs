use crate::Error;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fmt, str::FromStr};

/// SHA-256 identity with a canonical, validated wire representation.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ContentHash(String);

impl ContentHash {
    pub fn digest(bytes: impl AsRef<[u8]>) -> Self {
        Self(format!("sha256:{:x}", Sha256::digest(bytes.as_ref())))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl fmt::Display for ContentHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl FromStr for ContentHash {
    type Err = Error;
    fn from_str(value: &str) -> Result<Self, Error> {
        match value.strip_prefix("sha256:") {
            Some(hex)
                if hex.len() == 64
                    && hex
                        .bytes()
                        .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) =>
            {
                Ok(Self(value.into()))
            }
            _ => Err(Error::Invalid(
                "expected sha256 followed by 64 lowercase hexadecimal digits".into(),
            )),
        }
    }
}
impl TryFrom<String> for ContentHash {
    type Error = Error;
    fn try_from(value: String) -> Result<Self, Error> {
        value.parse()
    }
}
impl From<ContentHash> for String {
    fn from(value: ContentHash) -> Self {
        value.0
    }
}

/// Nonnegative additive extraction costs. Empty overrides preserve each original
/// egglog constructor's declared `:cost`, including the Record 004 packs.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CostModel {
    pub constructor_costs: BTreeMap<String, u64>,
}
impl CostModel {
    pub fn content_hash(&self) -> ContentHash {
        ContentHash::digest(
            serde_json::to_vec(self).expect("string/u64 map serialization is infallible"),
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RulePack {
    /// Exact program text identity; extraction additionally identifies its costs.
    pub id: ContentHash,
    pub program: String,
    pub cost_model: CostModel,
}
impl RulePack {
    pub fn new(program: impl Into<String>, cost_model: CostModel) -> Self {
        let program = program.into();
        Self {
            id: ContentHash::digest(program.as_bytes()),
            program,
            cost_model,
        }
    }
    pub fn validate(&self) -> Result<(), Error> {
        if self.id != ContentHash::digest(self.program.as_bytes()) {
            return Err(Error::Invalid(
                "rule pack identity does not match its exact program bytes".into(),
            ));
        }
        Ok(())
    }
}
