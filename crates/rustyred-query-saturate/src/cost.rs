use std::collections::BTreeMap;

use egg::{Analysis, CostFunction, DidMerge, EGraph, Id, Language};
use rustyred_thg_core::{
    AccessMethodRegistry, GraphStoreError, GraphStoreResult, LayerCardinality, LayerDirection,
    LayerDurability, LayerRegistry, Predicate, PredicateMode, QueryIr as QueryInput,
    RankingRegistry,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::lang::{
    AllRowsAttrs, EncodedAttr, ExpandAttrs, PredicateAttrs, QueryLang, RelationAttrs, RootAttrs,
};
use crate::rules::algebra_equivalents;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct LayerCostDescriptor {
    pub durability: LayerDurability,
    pub cardinality: LayerCardinality,
    pub direction: LayerDirection,
    pub pinnable: bool,
    pub unit_cost: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CostSnapshot {
    pub indexed_graph_version: u64,
    pub snapshot_ref: String,
    pub scan_costs: BTreeMap<String, f64>,
    pub rank_costs: BTreeMap<String, f64>,
    pub layer_descriptors: BTreeMap<String, LayerCostDescriptor>,
    pub combined_layer_penalty: f64,
}

impl CostSnapshot {
    pub fn capture(
        query: &QueryInput,
        access_methods: &AccessMethodRegistry,
        ranking_methods: &RankingRegistry,
        layers: &LayerRegistry,
    ) -> GraphStoreResult<Self> {
        let indexed_graph_version = access_methods.indexed_graph_version().ok_or_else(|| {
            GraphStoreError::new(
                "stale_cost_snapshot",
                "access methods have no current indexed graph version",
            )
        })?;
        let mut snapshot = Self {
            indexed_graph_version,
            snapshot_ref: String::new(),
            scan_costs: BTreeMap::new(),
            rank_costs: BTreeMap::new(),
            layer_descriptors: layers
                .layers()
                .map(|descriptor| {
                    let unit_cost = match descriptor.durability {
                        rustyred_thg_core::LayerDurability::Durable => 1.0,
                        rustyred_thg_core::LayerDurability::Derived => 1.25,
                        rustyred_thg_core::LayerDurability::Ephemeral => 1.5,
                    };
                    (
                        descriptor.id.to_string(),
                        LayerCostDescriptor {
                            durability: descriptor.durability,
                            cardinality: descriptor.cardinality,
                            direction: descriptor.direction,
                            pinnable: descriptor.pinnable,
                            unit_cost,
                        },
                    )
                })
                .collect(),
            combined_layer_penalty: 1.0,
        };

        for relation in &query.relations {
            for predicate in &relation.predicates {
                for candidate in algebra_equivalents(predicate) {
                    let key = predicate_cost_key(&relation.relation, &candidate)?;
                    if predicate_mode(&candidate) == PredicateMode::Rank {
                        let cost = ranking_methods
                            .methods()
                            .filter_map(|method| {
                                method
                                    .cost(&relation.relation, &candidate, None)
                                    .map(|cost| cost.rank_key())
                            })
                            .min_by(f64::total_cmp)
                            .unwrap_or(f64::INFINITY);
                        snapshot.rank_costs.insert(key, cost);
                    } else if !matches!(candidate, Predicate::Expand { .. }) {
                        let cost = access_methods
                            .methods()
                            .filter_map(|method| {
                                method
                                    .cost(&relation.relation, &candidate)
                                    .map(|cost| cost.rank_key())
                            })
                            .min_by(f64::total_cmp)
                            .unwrap_or(f64::INFINITY);
                        snapshot.scan_costs.insert(key, cost);
                    }
                }
            }
        }
        snapshot.snapshot_ref = snapshot.compute_ref()?;
        Ok(snapshot)
    }

    pub fn from_cost_maps(
        indexed_graph_version: u64,
        scan_costs: BTreeMap<String, f64>,
        rank_costs: BTreeMap<String, f64>,
        layer_costs: BTreeMap<String, f64>,
    ) -> GraphStoreResult<Self> {
        let mut snapshot = Self {
            indexed_graph_version,
            snapshot_ref: String::new(),
            scan_costs,
            rank_costs,
            layer_descriptors: layer_costs
                .into_iter()
                .map(|(id, unit_cost)| {
                    (
                        id,
                        LayerCostDescriptor {
                            durability: LayerDurability::Durable,
                            cardinality: LayerCardinality::Single,
                            direction: LayerDirection::Directed,
                            pinnable: true,
                            unit_cost,
                        },
                    )
                })
                .collect(),
            combined_layer_penalty: 1.0,
        };
        snapshot.snapshot_ref = snapshot.compute_ref()?;
        Ok(snapshot)
    }

    pub fn with_combined_layer_penalty(
        mut self,
        combined_layer_penalty: f64,
    ) -> GraphStoreResult<Self> {
        self.combined_layer_penalty = combined_layer_penalty.max(0.0);
        self.snapshot_ref = self.compute_ref()?;
        Ok(self)
    }

    fn compute_ref(&self) -> GraphStoreResult<String> {
        let payload = serde_json::to_vec(&(
            self.indexed_graph_version,
            &self.scan_costs,
            &self.rank_costs,
            &self.layer_descriptors,
            self.combined_layer_penalty.to_bits(),
        ))
        .map_err(|source| {
            GraphStoreError::new(
                "cost_snapshot_encode",
                format!("failed to encode cost snapshot: {source}"),
            )
        })?;
        Ok(format!("{:x}", Sha256::digest(payload)))
    }
}

pub fn ensure_snapshot_current(
    snapshot: &CostSnapshot,
    registry: &AccessMethodRegistry,
) -> GraphStoreResult<()> {
    let current = registry.indexed_graph_version();
    if current != Some(snapshot.indexed_graph_version) {
        return Err(GraphStoreError::new(
            "stale_cost_snapshot",
            format!(
                "cost snapshot is stamped at graph version {}, registry is at {:?}",
                snapshot.indexed_graph_version, current
            ),
        ));
    }
    Ok(())
}

pub fn predicate_cost_key(relation: &str, predicate: &Predicate) -> GraphStoreResult<String> {
    serde_json::to_string(&(relation, predicate)).map_err(|source| {
        GraphStoreError::new(
            "query_saturate_cost_key",
            format!("failed to encode predicate cost key: {source}"),
        )
    })
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct QueryAnalysisData {
    pub min_cost: f64,
    pub has_rank: bool,
    pub max_rank_layer_cardinality: usize,
}

#[derive(Clone, Debug, Default)]
pub struct SaturationAnalysis;

impl Analysis<QueryLang> for SaturationAnalysis {
    type Data = QueryAnalysisData;

    fn make(egraph: &mut EGraph<QueryLang, Self>, enode: &QueryLang) -> Self::Data {
        let mut data = QueryAnalysisData::default();
        for child in enode.children() {
            let child = &egraph[*child].data;
            data.min_cost += child.min_cost;
            data.has_rank |= child.has_rank;
            data.max_rank_layer_cardinality = data
                .max_rank_layer_cardinality
                .max(child.max_rank_layer_cardinality);
        }
        match enode {
            QueryLang::Rank(_) => {
                data.has_rank = true;
                data.max_rank_layer_cardinality = data
                    .max_rank_layer_cardinality
                    .max(rank_node_layer_cardinality(egraph, enode).unwrap_or(usize::MAX));
            }
            _ => {}
        }
        data
    }

    fn merge(&mut self, target: &mut Self::Data, source: Self::Data) -> DidMerge {
        let before = target.clone();
        target.min_cost = if target.min_cost == 0.0 {
            source.min_cost
        } else if source.min_cost == 0.0 {
            target.min_cost
        } else {
            target.min_cost.min(source.min_cost)
        };
        target.has_rank |= source.has_rank;
        target.max_rank_layer_cardinality = target
            .max_rank_layer_cardinality
            .max(source.max_rank_layer_cardinality);
        DidMerge(before != *target, true)
    }
}

#[derive(Clone, Debug)]
pub struct QueryCostFunction {
    pub snapshot: CostSnapshot,
}

impl CostFunction<QueryLang> for QueryCostFunction {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &QueryLang, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        match enode {
            QueryLang::Attr(attr) => self.attribute_cost(attr),
            QueryLang::Scan([relation, predicate]) => costs(*relation) + costs(*predicate) + 0.05,
            QueryLang::Expand([from, layers, dir]) => {
                costs(*from) + costs(*layers) + costs(*dir) + 0.2
            }
            QueryLang::Layer([layer]) => costs(*layer),
            QueryLang::Layers([head, rest]) => {
                let rest_cost = costs(*rest);
                costs(*head)
                    + rest_cost
                    + if rest_cost > 0.0 {
                        self.snapshot.combined_layer_penalty
                    } else {
                        0.0
                    }
            }
            QueryLang::LayersEmpty => 0.0,
            QueryLang::Intersect([left, right]) => costs(*left) + costs(*right) + 0.1,
            QueryLang::Union([left, right]) => costs(*left) + costs(*right) + 0.15,
            QueryLang::Rank([source, predicate, layer, k]) => {
                costs(*source) + costs(*predicate) + costs(*layer) + costs(*k) + 0.25
            }
            QueryLang::Limit([source, k]) => costs(*source) + costs(*k) + 0.02,
            QueryLang::Exists([q])
            | QueryLang::NotExists([q])
            | QueryLang::SemiJoin([q])
            | QueryLang::AntiJoin([q]) => costs(*q) + 0.05,
            QueryLang::QueryRoot([version, snapshot, body]) => {
                costs(*version) + costs(*snapshot) + costs(*body)
            }
        }
    }
}

impl QueryCostFunction {
    fn attribute_cost(&self, attr: &EncodedAttr) -> f64 {
        if let Ok(predicate) = attr.decode::<PredicateAttrs>() {
            let key = match predicate_cost_key(&predicate.relation.relation, &predicate.predicate) {
                Ok(key) => key,
                Err(_) => return f64::INFINITY,
            };
            return if predicate_mode(&predicate.predicate) == PredicateMode::Rank {
                self.snapshot
                    .rank_costs
                    .get(&key)
                    .copied()
                    .unwrap_or(f64::INFINITY)
            } else {
                self.snapshot
                    .scan_costs
                    .get(&key)
                    .copied()
                    .unwrap_or(f64::INFINITY)
            };
        }
        if let Ok(layer) = attr.decode::<rustyred_thg_core::LayerId>() {
            return self
                .snapshot
                .layer_descriptors
                .get(layer.as_str())
                .map(|descriptor| descriptor.unit_cost)
                .unwrap_or(1.0);
        }
        if attr.decode::<RootAttrs>().is_ok()
            || attr.decode::<RelationAttrs>().is_ok()
            || attr.decode::<ExpandAttrs>().is_ok()
            || attr.decode::<AllRowsAttrs>().is_ok()
            || attr.decode::<String>().is_ok()
            || attr.decode::<usize>().is_ok()
            || attr.decode::<Option<rustyred_thg_core::LayerId>>().is_ok()
            || attr.decode::<rustyred_thg_core::Direction>().is_ok()
        {
            return 0.0;
        }
        0.0
    }
}

pub fn rank_node_layer_cardinality(
    egraph: &EGraph<QueryLang, SaturationAnalysis>,
    node: &QueryLang,
) -> Option<usize> {
    let QueryLang::Rank([_, predicate_id, layer_id, _]) = node else {
        return None;
    };
    let predicate_cardinality = egraph[*predicate_id]
        .nodes
        .iter()
        .find_map(|node| match node {
            QueryLang::Attr(attr) => attr.decode::<PredicateAttrs>().ok().map(|attrs| match attrs
                .predicate
            {
                Predicate::Expand { layers, .. } => layers.len(),
                _ => 1,
            }),
            _ => None,
        });
    let structural_cardinality = egraph[*layer_id]
        .nodes
        .iter()
        .any(|node| matches!(node, QueryLang::Layer(_)))
        .then_some(1);
    predicate_cardinality.or(structural_cardinality)
}

fn predicate_mode(predicate: &Predicate) -> PredicateMode {
    match predicate {
        Predicate::Knn { .. } => PredicateMode::Rank,
        Predicate::TextMatch { mode, .. } | Predicate::Expand { mode, .. } => *mode,
        _ => PredicateMode::Filter,
    }
}
