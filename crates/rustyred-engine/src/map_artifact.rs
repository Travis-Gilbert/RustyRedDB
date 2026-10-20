use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::context_view::{FreshnessStatus, HydrationHandle};
use crate::query_receipt::ReceiptScope;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MapArtifactType {
    Codebase,
    UserMemory,
    Rule,
    Tool,
    Skill,
    Project,
    Domain,
    Run,
    Training,
    Adapter,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct MapSection {
    pub id: String,
    pub title: String,
    pub summary: String,
    #[serde(default)]
    pub hydration_handles: Vec<HydrationHandle>,
    #[serde(default)]
    pub source_atom_ids: Vec<String>,
    #[serde(default)]
    pub positive_label_ids: Vec<String>,
    #[serde(default)]
    pub negative_label_ids: Vec<String>,
    pub usage_count: u64,
    pub outcome_score: i64,
    pub freshness_status: FreshnessStatus,
}

impl MapSection {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            summary: summary.into(),
            hydration_handles: Vec::new(),
            source_atom_ids: Vec::new(),
            positive_label_ids: Vec::new(),
            negative_label_ids: Vec::new(),
            usage_count: 0,
            outcome_score: 0,
            freshness_status: FreshnessStatus::Fresh,
        }
    }

    pub fn record_usage(&mut self, positive: bool, label_id: impl Into<String>) {
        self.usage_count = self.usage_count.saturating_add(1);
        if positive {
            self.outcome_score = self.outcome_score.saturating_add(1);
            self.positive_label_ids.push(label_id.into());
        } else {
            self.outcome_score = self.outcome_score.saturating_sub(1);
            self.negative_label_ids.push(label_id.into());
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct MapArtifact {
    pub id: String,
    pub map_type: MapArtifactType,
    pub scope: ReceiptScope,
    pub graph_version: u64,
    pub version: u64,
    pub freshness_status: FreshnessStatus,
    pub reuse_score: f64,
    #[serde(default)]
    pub sections: Vec<MapSection>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MapArtifactDiff {
    pub from_map_id: String,
    pub to_map_id: String,
    pub from_version: u64,
    pub to_version: u64,
    #[serde(default)]
    pub added_section_ids: Vec<String>,
    #[serde(default)]
    pub removed_section_ids: Vec<String>,
    #[serde(default)]
    pub changed_section_ids: Vec<String>,
    #[serde(default)]
    pub stale_section_ids: Vec<String>,
}

impl MapArtifact {
    pub fn new(
        id: impl Into<String>,
        map_type: MapArtifactType,
        scope: ReceiptScope,
        graph_version: u64,
    ) -> Self {
        Self {
            id: id.into(),
            map_type,
            scope,
            graph_version,
            version: 1,
            freshness_status: FreshnessStatus::Fresh,
            reuse_score: 0.0,
            sections: Vec::new(),
        }
    }

    pub fn add_section(&mut self, section: MapSection) {
        self.sections.push(section);
    }

    pub fn section(&self, id: &str) -> Option<&MapSection> {
        self.sections.iter().find(|section| section.id == id)
    }

    pub fn section_mut(&mut self, id: &str) -> Option<&mut MapSection> {
        self.sections.iter_mut().find(|section| section.id == id)
    }

    pub fn record_section_usage(&mut self, section_id: &str, positive: bool, label_id: String) {
        if let Some(section) = self.section_mut(section_id) {
            section.record_usage(positive, label_id);
            let total_score = self
                .sections
                .iter()
                .map(|section| section.outcome_score)
                .sum::<i64>();
            self.reuse_score = total_score as f64;
        }
    }

    pub fn diff_against(&self, next: &MapArtifact) -> MapArtifactDiff {
        let current = sections_by_id(&self.sections);
        let regenerated = sections_by_id(&next.sections);
        let current_ids = current.keys().cloned().collect::<BTreeSet<_>>();
        let regenerated_ids = regenerated.keys().cloned().collect::<BTreeSet<_>>();

        let mut added_section_ids = regenerated_ids
            .difference(&current_ids)
            .cloned()
            .collect::<Vec<_>>();
        let mut removed_section_ids = current_ids
            .difference(&regenerated_ids)
            .cloned()
            .collect::<Vec<_>>();
        let mut changed_section_ids = current_ids
            .intersection(&regenerated_ids)
            .filter_map(|id| {
                let left = current.get(id)?;
                let right = regenerated.get(id)?;
                section_content_changed(left, right).then(|| id.clone())
            })
            .collect::<Vec<_>>();
        let mut stale_section_ids = self
            .sections
            .iter()
            .filter(|section| section.freshness_status != FreshnessStatus::Fresh)
            .map(|section| section.id.clone())
            .collect::<Vec<_>>();

        added_section_ids.sort();
        removed_section_ids.sort();
        changed_section_ids.sort();
        stale_section_ids.sort();

        MapArtifactDiff {
            from_map_id: self.id.clone(),
            to_map_id: next.id.clone(),
            from_version: self.version,
            to_version: next.version,
            added_section_ids,
            removed_section_ids,
            changed_section_ids,
            stale_section_ids,
        }
    }

    pub fn mark_stale_if_graph_version_behind(&mut self, current_graph_version: u64) -> bool {
        if self.graph_version >= current_graph_version {
            return false;
        }
        self.freshness_status = FreshnessStatus::NeedsRebuild;
        for section in &mut self.sections {
            if section
                .hydration_handles
                .iter()
                .any(|handle| handle.graph_version < current_graph_version)
            {
                section.freshness_status = FreshnessStatus::Stale;
            }
        }
        true
    }

    pub fn refresh_from(&mut self, mut regenerated: MapArtifact) -> MapArtifactDiff {
        regenerated.id = self.id.clone();
        regenerated.version = self.version.saturating_add(1);
        regenerated.freshness_status = FreshnessStatus::Fresh;
        for section in &mut regenerated.sections {
            section.freshness_status = FreshnessStatus::Fresh;
        }
        let diff = self.diff_against(&regenerated);
        *self = regenerated;
        diff
    }
}

fn sections_by_id(sections: &[MapSection]) -> BTreeMap<String, &MapSection> {
    sections
        .iter()
        .map(|section| (section.id.clone(), section))
        .collect()
}

fn section_content_changed(left: &MapSection, right: &MapSection) -> bool {
    left.title != right.title
        || left.summary != right.summary
        || left.hydration_handles != right.hydration_handles
        || left.source_atom_ids != right.source_atom_ids
}
