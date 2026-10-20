//! Structured narrative views over existing graph claims.
//!
//! A narrative is a read model: it orders existing events, links them through
//! cited causal evidence, and produces a readable arc for product surfaces. It
//! does not create new truth claims.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::state::stable_hash;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NarrativeEvent {
    pub event_id: String,
    pub claim_node_id: String,
    pub title: String,
    pub occurred_at_ms: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NarrativeLink {
    pub from_event_id: String,
    pub to_event_id: String,
    pub relation: String,
    pub evidence_node_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CommonPlaceArcItem {
    pub position: usize,
    pub event_id: String,
    pub text: String,
    pub referenced_claim_node_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NarrativeView {
    pub narrative_id: String,
    pub title: String,
    pub events: Vec<NarrativeEvent>,
    pub causal_links: Vec<NarrativeLink>,
    pub constituent_node_ids: Vec<String>,
    pub asserts_new_claims: bool,
    pub commonplace_arc: Vec<CommonPlaceArcItem>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NarrativeBuildError {
    pub code: String,
    pub message: String,
}

impl NarrativeBuildError {
    fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

pub fn build_narrative_view(
    title: impl Into<String>,
    events: Vec<NarrativeEvent>,
    links: Vec<NarrativeLink>,
) -> Result<NarrativeView, NarrativeBuildError> {
    if events.is_empty() {
        return Err(NarrativeBuildError::new(
            "empty_narrative",
            "narrative requires at least one event",
        ));
    }
    let event_ids = events
        .iter()
        .map(|event| event.event_id.clone())
        .collect::<BTreeSet<_>>();
    let mut constituent_node_ids = events
        .iter()
        .map(|event| event.claim_node_id.clone())
        .collect::<BTreeSet<_>>();
    for link in &links {
        if !event_ids.contains(&link.from_event_id) || !event_ids.contains(&link.to_event_id) {
            return Err(NarrativeBuildError::new(
                "dangling_narrative_link",
                format!(
                    "link {} -> {} references a missing event",
                    link.from_event_id, link.to_event_id
                ),
            ));
        }
        constituent_node_ids.insert(link.evidence_node_id.clone());
    }
    let mut events = events;
    events.sort_by(|left, right| {
        left.occurred_at_ms
            .cmp(&right.occurred_at_ms)
            .then_with(|| left.event_id.cmp(&right.event_id))
    });
    let mut links = links;
    links.sort_by(|left, right| {
        left.to_event_id
            .cmp(&right.to_event_id)
            .then_with(|| left.from_event_id.cmp(&right.from_event_id))
            .then_with(|| left.relation.cmp(&right.relation))
    });
    let title = title.into();
    let commonplace_arc = events
        .iter()
        .enumerate()
        .map(|(index, event)| CommonPlaceArcItem {
            position: index,
            event_id: event.event_id.clone(),
            text: event.title.clone(),
            referenced_claim_node_id: event.claim_node_id.clone(),
        })
        .collect::<Vec<_>>();
    let constituent_node_ids = constituent_node_ids.into_iter().collect::<Vec<_>>();
    Ok(NarrativeView {
        narrative_id: narrative_id(&title, &events, &links)?,
        title,
        events,
        causal_links: links,
        constituent_node_ids,
        asserts_new_claims: false,
        commonplace_arc,
    })
}

fn narrative_id(
    title: &str,
    events: &[NarrativeEvent],
    links: &[NarrativeLink],
) -> Result<String, NarrativeBuildError> {
    Ok(format!(
        "narrative:{}",
        stable_hash(json!({
            "title": title,
            "events": events.iter().map(|event| &event.event_id).collect::<Vec<_>>(),
            "links": links,
        }))
        .map_err(|error| {
            NarrativeBuildError::new(
                "narrative_hash_failed",
                format!("{}: {}", error.code, error.message),
            )
        })?
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrative_orders_events_and_references_existing_claims_without_asserting_new_ones() {
        let events = vec![
            NarrativeEvent {
                event_id: "event:mitigation".to_string(),
                claim_node_id: "claim:mitigation".to_string(),
                title: "Retry window reduced queue pressure".to_string(),
                occurred_at_ms: 300,
            },
            NarrativeEvent {
                event_id: "event:incident".to_string(),
                claim_node_id: "claim:incident".to_string(),
                title: "Scheduler contention spiked".to_string(),
                occurred_at_ms: 100,
            },
            NarrativeEvent {
                event_id: "event:hypothesis".to_string(),
                claim_node_id: "claim:hypothesis".to_string(),
                title: "Orbit analogy suggested lock-ring contention".to_string(),
                occurred_at_ms: 200,
            },
        ];
        let links = vec![
            NarrativeLink {
                from_event_id: "event:incident".to_string(),
                to_event_id: "event:hypothesis".to_string(),
                relation: "motivated".to_string(),
                evidence_node_id: "claim:incident".to_string(),
            },
            NarrativeLink {
                from_event_id: "event:hypothesis".to_string(),
                to_event_id: "event:mitigation".to_string(),
                relation: "informed".to_string(),
                evidence_node_id: "claim:hypothesis".to_string(),
            },
        ];

        let narrative =
            build_narrative_view("Scheduler contention arc", events, links).expect("narrative");

        assert_eq!(
            narrative
                .events
                .iter()
                .map(|event| event.event_id.as_str())
                .collect::<Vec<_>>(),
            vec!["event:incident", "event:hypothesis", "event:mitigation"]
        );
        assert!(!narrative.asserts_new_claims);
        for link in &narrative.causal_links {
            assert!(narrative
                .events
                .iter()
                .any(|event| event.event_id == link.from_event_id));
            assert!(narrative
                .events
                .iter()
                .any(|event| event.event_id == link.to_event_id));
            assert!(narrative
                .constituent_node_ids
                .contains(&link.evidence_node_id));
        }
        assert_eq!(narrative.commonplace_arc.len(), 3);
        assert_eq!(narrative.commonplace_arc[0].position, 0);
        assert_eq!(
            narrative.commonplace_arc[0].referenced_claim_node_id,
            "claim:incident"
        );
        assert!(narrative.commonplace_arc[0]
            .text
            .contains("Scheduler contention"));
    }
}
