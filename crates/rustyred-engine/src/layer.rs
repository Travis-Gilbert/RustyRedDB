//! Multiplex layer registry: named relations over one node set.
//!
//! Layers are registered and looked up. Call sites decide behavior from
//! descriptors, not from matching `edge_type` string literals. See
//! `docs/plans/SPEC-MULTIPLEX-LAYERS.md`.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::graph::error::{GraphStoreError, GraphStoreResult};

/// Newtype identity for a multiplex layer.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct LayerId(String);

impl LayerId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for LayerId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<String> for LayerId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl AsRef<str> for LayerId {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl fmt::Display for LayerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerDurability {
    Durable,
    Derived,
    Ephemeral,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerCardinality {
    Single,
    Multi,
    Ranked,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerDirection {
    Directed,
    Symmetric,
    Positional,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LayerDescriptor {
    pub id: LayerId,
    pub durability: LayerDurability,
    pub cardinality: LayerCardinality,
    pub direction: LayerDirection,
    /// Explicit pin safety. Not derived from durability: a derived layer whose
    /// producer is nondeterministic across time is not pinnable.
    pub pinnable: bool,
}

/// Registry of multiplex layers. Mirrors the `RankingRegistry` construction
/// idiom: `new`, `register`, ordered iteration; lookup by id is an extra.
#[derive(Clone, Debug, Default)]
pub struct LayerRegistry {
    by_id: BTreeMap<LayerId, LayerDescriptor>,
    order: Vec<LayerId>,
}

impl LayerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, descriptor: LayerDescriptor) {
        let id = descriptor.id.clone();
        if self.by_id.insert(id.clone(), descriptor).is_none() {
            self.order.push(id);
        }
    }

    pub fn get(&self, id: &LayerId) -> Option<&LayerDescriptor> {
        self.by_id.get(id)
    }

    pub fn layers(&self) -> impl Iterator<Item = &LayerDescriptor> {
        self.order.iter().filter_map(|id| self.by_id.get(id))
    }

    /// Seven native inventory layers from SPEC-MULTIPLEX-LAYERS.
    pub fn with_native_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(LayerDescriptor {
            id: LayerId::from("composition"),
            durability: LayerDurability::Durable,
            cardinality: LayerCardinality::Single,
            direction: LayerDirection::Directed,
            pinnable: true,
        });
        registry.register(LayerDescriptor {
            id: LayerId::from("derivation"),
            durability: LayerDurability::Derived,
            cardinality: LayerCardinality::Single,
            direction: LayerDirection::Directed,
            pinnable: true,
        });
        registry.register(LayerDescriptor {
            id: LayerId::from("temporal"),
            durability: LayerDurability::Durable,
            cardinality: LayerCardinality::Single,
            direction: LayerDirection::Directed,
            pinnable: true,
        });
        registry.register(LayerDescriptor {
            id: LayerId::from("authority"),
            durability: LayerDurability::Durable,
            cardinality: LayerCardinality::Single,
            direction: LayerDirection::Directed,
            pinnable: true,
        });
        registry.register(LayerDescriptor {
            id: LayerId::from("annotation"),
            durability: LayerDurability::Durable,
            cardinality: LayerCardinality::Multi,
            direction: LayerDirection::Directed,
            pinnable: true,
        });
        registry.register(LayerDescriptor {
            id: LayerId::from("presence"),
            durability: LayerDurability::Ephemeral,
            cardinality: LayerCardinality::Single,
            direction: LayerDirection::Symmetric,
            pinnable: false,
        });
        registry.register(LayerDescriptor {
            id: LayerId::from("semantic"),
            durability: LayerDurability::Derived,
            cardinality: LayerCardinality::Ranked,
            direction: LayerDirection::Symmetric,
            pinnable: false,
        });
        registry
    }

    /// Refuse when any requested layer is missing or not pinnable.
    pub fn check_pinnable(&self, layers: &[LayerId]) -> GraphStoreResult<()> {
        for layer in layers {
            match self.get(layer) {
                None => {
                    return Err(GraphStoreError::new(
                        "layer_not_registered",
                        format!("layer `{layer}` is not registered"),
                    ));
                }
                Some(descriptor) if !descriptor.pinnable => {
                    return Err(GraphStoreError::new(
                        "layer_not_pinnable",
                        format!(
                            "layer `{layer}` is not pinnable; version-pinned queries cannot include it"
                        ),
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }

    /// Default selection layer set: every registered layer.
    pub fn all_layer_ids(&self) -> Vec<LayerId> {
        self.order.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_and_lookup_round_trip() {
        let mut registry = LayerRegistry::new();
        registry.register(LayerDescriptor {
            id: LayerId::from("composition"),
            durability: LayerDurability::Durable,
            cardinality: LayerCardinality::Single,
            direction: LayerDirection::Directed,
            pinnable: true,
        });
        let got = registry
            .get(&LayerId::from("composition"))
            .expect("present");
        assert_eq!(got.id.as_str(), "composition");
        assert!(got.pinnable);
        assert!(registry.get(&LayerId::from("missing")).is_none());
    }

    #[test]
    fn native_defaults_are_exactly_seven_with_presence_and_annotation_flags() {
        let registry = LayerRegistry::with_native_defaults();
        let layers: Vec<_> = registry.layers().collect();
        assert_eq!(layers.len(), 7);
        let presence = registry.get(&LayerId::from("presence")).expect("presence");
        assert_eq!(presence.durability, LayerDurability::Ephemeral);
        assert!(!presence.pinnable);
        let annotation = registry
            .get(&LayerId::from("annotation"))
            .expect("annotation");
        assert_eq!(annotation.cardinality, LayerCardinality::Multi);
        assert!(annotation.pinnable);
        let semantic = registry.get(&LayerId::from("semantic")).expect("semantic");
        assert_eq!(semantic.cardinality, LayerCardinality::Ranked);
        assert!(!semantic.pinnable);
    }

    #[test]
    fn check_pinnable_refuses_presence() {
        let registry = LayerRegistry::with_native_defaults();
        let err = registry
            .check_pinnable(&[LayerId::from("presence")])
            .expect_err("presence not pinnable");
        assert_eq!(err.code, "layer_not_pinnable");
        assert!(err.message.contains("presence"));
        registry
            .check_pinnable(&[LayerId::from("composition"), LayerId::from("authority")])
            .expect("durable layers pinnable");
    }
}
