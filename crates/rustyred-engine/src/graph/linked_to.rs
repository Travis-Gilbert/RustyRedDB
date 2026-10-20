//! Canonical link creation-version normalization at the actual writer boundary.
use crate::{EdgeRecord, GraphStoreError, GraphStoreResult, Hash};
use theorem_surface_contracts::link::{linked_to_identity_bytes, LinkedToRecord};
pub use theorem_surface_contracts::link::{LINKED_TO_PROPERTIES, LINKED_TO_SCHEMA, LINKED_TO_TYPE};

pub(crate) fn stamp_first_version(
    edge: &mut EdgeRecord,
    prior: Option<&EdgeRecord>,
    allocated_version: u64,
) -> GraphStoreResult<()> {
    if edge.edge_type != LINKED_TO_TYPE {
        return Ok(());
    }
    let invalid = || {
        GraphStoreError::new(
            "invalid_linked_to_record",
            "canonical LINKED_TO binding or first write version is invalid",
        )
    };
    let record: LinkedToRecord<serde_json::Value> =
        serde_json::from_value(edge.properties.clone()).map_err(|_| invalid())?;
    record.validate().map_err(|_| invalid())?;
    let expected_id = format!(
        "linked-to:{}",
        Hash::of(&linked_to_identity_bytes(
            &edge.from_id,
            &edge.to_id,
            record.kind,
        ))
        .to_prefixed()
    );
    if edge.id != expected_id || !record.visibility.is_object() || allocated_version == 0 {
        return Err(invalid());
    }
    let first_version = match prior {
        Some(prior) => {
            let old: LinkedToRecord<serde_json::Value> =
                serde_json::from_value(prior.properties.clone()).map_err(|_| invalid())?;
            old.validate().map_err(|_| invalid())?;
            let first = old.written_at_graph_version.ok_or_else(invalid)?;
            if prior.edge_type != LINKED_TO_TYPE
                || prior.from_id != edge.from_id
                || prior.to_id != edge.to_id
                || old.kind != record.kind
                || first > prior.version
                || first >= allocated_version
                || record
                    .written_at_graph_version
                    .is_some_and(|value| value != first)
            {
                return Err(invalid());
            }
            first
        }
        None => {
            if record
                .written_at_graph_version
                .is_some_and(|value| value != allocated_version)
            {
                return Err(invalid());
            }
            allocated_version
        }
    };
    edge.properties["written_at_graph_version"] = serde_json::json!(first_version);
    Ok(())
}
