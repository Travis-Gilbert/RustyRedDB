//! Support multiplicity for derived facts (Z-set annotations).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Signed multiplicity of a fact in the derived set. Negative values are
/// legal in transit and are an error at rest.
pub type Multiplicity = i64;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SupportedFact {
    pub fact_id: String,
    pub multiplicity: Multiplicity,
    pub base: bool,
    pub justifications: Vec<Justification>,
    /// Saturation relation / Datalog predicate. Hashed native fact ids do not
    /// parse as `predicate(args)`, so distinct/DRed selection keys off this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Justification {
    pub rule_id: String,
    pub premises: Vec<String>,
}

impl SupportedFact {
    pub fn base(fact_id: impl Into<String>) -> Self {
        Self {
            fact_id: fact_id.into(),
            multiplicity: 1,
            base: true,
            justifications: Vec::new(),
            relation: None,
        }
    }

    pub fn is_present(&self) -> bool {
        self.multiplicity > 0
    }
}

/// Clamp a Z-set to set semantics: any positive multiplicity becomes 1,
/// any non-positive becomes 0 (and is removed from the map).
pub fn distinct(facts: &mut BTreeMap<String, Multiplicity>) {
    let keys: Vec<String> = facts.keys().cloned().collect();
    for key in keys {
        match facts.get(&key).copied().unwrap_or(0) {
            m if m > 0 => {
                facts.insert(key, 1);
            }
            _ => {
                facts.remove(&key);
            }
        }
    }
}

/// Incremental form. Emits a change only when a multiplicity crosses the
/// zero boundary: +1 on a transition from non-positive to positive,
/// -1 on the reverse, nothing otherwise.
pub fn distinct_delta(
    previous: &BTreeMap<String, Multiplicity>,
    next: &BTreeMap<String, Multiplicity>,
) -> BTreeMap<String, Multiplicity> {
    let mut out = BTreeMap::new();
    let keys: std::collections::BTreeSet<&String> = previous.keys().chain(next.keys()).collect();
    for key in keys {
        let prev = previous.get(key).copied().unwrap_or(0);
        let nxt = next.get(key).copied().unwrap_or(0);
        let prev_pos = prev > 0;
        let next_pos = nxt > 0;
        if !prev_pos && next_pos {
            out.insert(key.clone(), 1);
        } else if prev_pos && !next_pos {
            out.insert(key.clone(), -1);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_clamps_and_drops_non_positive() {
        let mut facts = BTreeMap::from([
            ("a".to_string(), 3),
            ("b".to_string(), 0),
            ("c".to_string(), -2),
        ]);
        distinct(&mut facts);
        assert_eq!(facts, BTreeMap::from([("a".to_string(), 1)]));
    }

    #[test]
    fn distinct_delta_is_nonlinear_at_zero_boundary() {
        let previous = BTreeMap::from([("a".to_string(), 1)]);
        let next_same_side = BTreeMap::from([("a".to_string(), 4)]);
        assert!(distinct_delta(&previous, &next_same_side).is_empty());

        let next_zero = BTreeMap::from([("a".to_string(), 0)]);
        assert_eq!(
            distinct_delta(&previous, &next_zero),
            BTreeMap::from([("a".to_string(), -1)])
        );
    }
}
