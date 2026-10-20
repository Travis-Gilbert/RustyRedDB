//! Assumption-based truth maintenance labels for agency and derivation work.
//!
//! A label is minimal DNF: each inner set is one sufficient assumption
//! environment and the outer set is the set of alternative environments.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub type AssumptionId = String;
pub type AssumptionEnvironment = BTreeSet<AssumptionId>;

/// Boolean-semiring provenance represented as minimal DNF.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct BooleanProvenance {
    pub environments: BTreeSet<AssumptionEnvironment>,
}

impl BooleanProvenance {
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn one() -> Self {
        Self {
            environments: [AssumptionEnvironment::new()].into_iter().collect(),
        }
    }

    pub fn environment(environment: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            environments: [environment.into_iter().map(Into::into).collect()]
                .into_iter()
                .collect(),
        }
        .minimized()
    }

    /// Addition: union alternate sufficient environments and remove supersets.
    pub fn aggregate(&self, other: &Self) -> Self {
        Self {
            environments: self
                .environments
                .union(&other.environments)
                .cloned()
                .collect(),
        }
        .minimized()
    }

    /// Multiplication: pairwise union requirements and remove supersets.
    pub fn combine(&self, other: &Self) -> Self {
        let environments = self
            .environments
            .iter()
            .flat_map(|left| {
                other.environments.iter().map(move |right| {
                    left.union(right)
                        .cloned()
                        .collect::<AssumptionEnvironment>()
                })
            })
            .collect();
        Self { environments }.minimized()
    }

    pub fn minimized(&self) -> Self {
        let environments = self
            .environments
            .iter()
            .filter(|candidate| {
                !self
                    .environments
                    .iter()
                    .any(|other| other != *candidate && other.is_subset(candidate))
            })
            .cloned()
            .collect();
        Self { environments }
    }

    pub fn with_nogoods(&self, nogoods: &[AssumptionEnvironment]) -> Self {
        Self {
            environments: self
                .environments
                .iter()
                .filter(|environment| environment_is_valid(environment, nogoods))
                .cloned()
                .collect(),
        }
        .minimized()
    }

    /// Query a label under a context where one assumption is unavailable.
    ///
    /// This is deliberately a filter rather than a label-maintenance operation:
    /// retraction does not create a new justification or undo a nogood, so it
    /// must not recalculate or mutate the stored ATMS label.
    pub fn without_assumption(&self, assumption_id: &str) -> Self {
        Self {
            environments: self
                .environments
                .iter()
                .filter(|environment| !environment.contains(assumption_id))
                .cloned()
                .collect(),
        }
        .minimized()
    }
}

/// An environment is invalid when it contains a recorded nogood as a subset.
pub fn environment_is_valid(
    environment: &AssumptionEnvironment,
    nogoods: &[AssumptionEnvironment],
) -> bool {
    !nogoods.iter().any(|nogood| nogood.is_subset(environment))
}

/// Store only minimal nogoods: a superset adds no new exclusion power.
pub fn minimize_nogoods(
    nogoods: impl IntoIterator<Item = AssumptionEnvironment>,
) -> Vec<AssumptionEnvironment> {
    let values = nogoods.into_iter().collect::<BTreeSet<_>>();
    values
        .iter()
        .filter(|candidate| {
            !values
                .iter()
                .any(|other| other != *candidate && other.is_subset(candidate))
        })
        .cloned()
        .collect()
}

/// Durable calculated-label payload. `complete = false` means this is a
/// retained frontier, not a claim to enumerate every possible strategy.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AtmsLabelRecord {
    pub value_id: String,
    pub environments: BTreeSet<AssumptionEnvironment>,
    pub source_graph_version: u64,
    pub nogood_version: u64,
    pub complete: bool,
    pub pruned_environment_count: usize,
}

impl AtmsLabelRecord {
    pub fn bounded(
        value_id: impl Into<String>,
        label: &BooleanProvenance,
        nogoods: &[AssumptionEnvironment],
        source_graph_version: u64,
        nogood_version: u64,
        max_environment_width: usize,
        max_environments: usize,
    ) -> Self {
        let mut pruned_environment_count = 0;
        let mut kept = label
            .environments
            .iter()
            .filter_map(|environment| {
                if environment.len() > max_environment_width
                    || !environment_is_valid(environment, nogoods)
                {
                    pruned_environment_count += 1;
                    None
                } else {
                    Some(environment.clone())
                }
            })
            .collect::<Vec<_>>();
        kept.sort_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)));
        if kept.len() > max_environments {
            pruned_environment_count += kept.len() - max_environments;
            kept.truncate(max_environments);
        }
        Self {
            value_id: value_id.into(),
            environments: BooleanProvenance {
                environments: kept.into_iter().collect(),
            }
            .minimized()
            .environments,
            source_graph_version,
            nogood_version,
            complete: pruned_environment_count == 0,
            pruned_environment_count,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AtmsRecalculation {
    pub changed_assumption_id: AssumptionId,
    pub recalculated_value_ids: Vec<String>,
    pub materially_changed_value_ids: Vec<String>,
}

/// Dependency-indexed label cache. Justification and nogood changes update
/// dependent labels; context retraction remains a query-time filter.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct IncrementalAtms {
    pub labels: BTreeMap<String, AtmsLabelRecord>,
    pub dependency_index: BTreeMap<AssumptionId, BTreeSet<String>>,
}

impl IncrementalAtms {
    pub fn upsert_label(&mut self, label: AtmsLabelRecord) {
        self.remove_dependencies_for(&label.value_id);
        for environment in &label.environments {
            for assumption_id in environment {
                self.dependency_index
                    .entry(assumption_id.clone())
                    .or_default()
                    .insert(label.value_id.clone());
            }
        }
        self.labels.insert(label.value_id.clone(), label);
    }

    pub fn recalculate_after_label_change(
        &mut self,
        changed_assumption_id: impl Into<String>,
        mut calculate: impl FnMut(&str) -> AtmsLabelRecord,
    ) -> AtmsRecalculation {
        let changed_assumption_id = changed_assumption_id.into();
        let value_ids = self
            .dependency_index
            .get(&changed_assumption_id)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect::<Vec<_>>();
        let mut materially_changed_value_ids = Vec::new();
        for value_id in &value_ids {
            let next = calculate(value_id);
            if self.labels.get(value_id) != Some(&next) {
                materially_changed_value_ids.push(value_id.clone());
            }
            self.upsert_label(next);
        }
        AtmsRecalculation {
            changed_assumption_id,
            recalculated_value_ids: value_ids,
            materially_changed_value_ids,
        }
    }

    /// Read a stored label under a context where one assumption is absent.
    /// This never mutates the cache or invokes a label calculator.
    pub fn label_without_assumption(
        &self,
        value_id: &str,
        assumption_id: &str,
    ) -> Option<BooleanProvenance> {
        self.labels
            .get(value_id)
            .map(|label| BooleanProvenance {
                environments: label.environments.clone(),
            })
            .map(|label| label.without_assumption(assumption_id))
    }

    fn remove_dependencies_for(&mut self, value_id: &str) {
        for values in self.dependency_index.values_mut() {
            values.remove(value_id);
        }
        self.dependency_index.retain(|_, values| !values.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nogoods_remove_every_superset_and_minimize_their_storage() {
        let label = BooleanProvenance {
            environments: [
                ["A".to_string()].into_iter().collect(),
                ["B".to_string()].into_iter().collect(),
                ["A".to_string(), "B".to_string()].into_iter().collect(),
                ["A".to_string(), "B".to_string(), "C".to_string()]
                    .into_iter()
                    .collect(),
            ]
            .into_iter()
            .collect(),
        };
        let nogoods = minimize_nogoods([
            ["A".to_string(), "B".to_string()].into_iter().collect(),
            ["A".to_string(), "B".to_string(), "C".to_string()]
                .into_iter()
                .collect(),
        ]);
        let retained = label.with_nogoods(&nogoods);
        assert_eq!(nogoods.len(), 1);
        assert_eq!(retained.environments.len(), 2);
        assert!(retained
            .environments
            .iter()
            .all(|environment| environment_is_valid(environment, &nogoods)));
    }

    #[test]
    fn bounded_records_name_an_incomplete_frontier_and_recalculate_only_dependents() {
        let label = BooleanProvenance {
            environments: [
                ["A".to_string()].into_iter().collect(),
                ["B".to_string()].into_iter().collect(),
            ]
            .into_iter()
            .collect(),
        };
        let record = AtmsLabelRecord::bounded("value:one", &label, &[], 7, 3, 2, 1);
        assert!(!record.complete);
        assert_eq!(record.pruned_environment_count, 1);

        let mut index = IncrementalAtms::default();
        index.upsert_label(record.clone());
        index.upsert_label(AtmsLabelRecord::bounded(
            "value:two",
            &BooleanProvenance::environment(["C"]),
            &[],
            7,
            3,
            2,
            8,
        ));
        let recalculation = index.recalculate_after_label_change("A", |value_id| {
            AtmsLabelRecord::bounded(value_id, &BooleanProvenance::zero(), &[], 8, 3, 2, 8)
        });
        assert_eq!(recalculation.recalculated_value_ids, vec!["value:one"]);
        assert_eq!(
            recalculation.materially_changed_value_ids,
            vec!["value:one"]
        );
    }

    #[test]
    fn retraction_is_a_query_time_context_filter() {
        let mut index = IncrementalAtms::default();
        index.upsert_label(AtmsLabelRecord::bounded(
            "value:delivery",
            &BooleanProvenance {
                environments: [
                    ["vendor".to_string(), "contract".to_string()]
                        .into_iter()
                        .collect(),
                    ["fallback".to_string()].into_iter().collect(),
                ]
                .into_iter()
                .collect(),
            },
            &[],
            7,
            3,
            4,
            8,
        ));

        let cached_before = index.labels["value:delivery"].environments.clone();
        let filtered = index
            .label_without_assumption("value:delivery", "vendor")
            .expect("stored label");
        assert_eq!(filtered.environments.len(), 1);
        assert!(filtered.environments.contains(
            &["fallback".to_string()]
                .into_iter()
                .collect::<AssumptionEnvironment>()
        ));
        assert_eq!(index.labels["value:delivery"].environments, cached_before);
    }
}
