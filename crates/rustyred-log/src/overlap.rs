//! Canonical read/write-set overlap algebra.
//!
//! Every consumer (OCC, invalidation, subscriptions, and fold hotness) uses
//! [`read_write_overlap`]. Sets normalize at construction and deserialization,
//! so equivalent inputs have one representation before the overlap walk runs.

use serde::{de, Deserialize, Deserializer, Serialize};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;

/// One addressable portion of derived state.
///
/// Intervals are half-open: `start` is included and `end` is excluded.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub enum AccessPath {
    Wildcard,
    Entity {
        entity: String,
    },
    Attribute {
        entity: String,
        attribute: String,
    },
    Interval {
        entity: String,
        attribute: String,
        start: i64,
        end: i64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OverlapError {
    EmptyIdentifier(&'static str),
    EmptyInterval { start: i64, end: i64 },
}

impl fmt::Display for OverlapError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(field) => write!(formatter, "{field} must not be empty"),
            Self::EmptyInterval { start, end } => write!(
                formatter,
                "interval must be non-empty and half-open: start {start}, end {end}"
            ),
        }
    }
}

impl Error for OverlapError {}

macro_rules! access_set {
    ($name:ident) => {
        #[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
        #[serde(transparent)]
        pub struct $name(Vec<AccessPath>);

        impl $name {
            pub fn new(paths: impl IntoIterator<Item = AccessPath>) -> Result<Self, OverlapError> {
                normalize(paths).map(Self)
            }

            pub fn entries(&self) -> &[AccessPath] {
                &self.0
            }

            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let paths = Vec::<AccessPath>::deserialize(deserializer)?;
                Self::new(paths).map_err(de::Error::custom)
            }
        }

        impl TryFrom<Vec<AccessPath>> for $name {
            type Error = OverlapError;

            fn try_from(paths: Vec<AccessPath>) -> Result<Self, Self::Error> {
                Self::new(paths)
            }
        }
    };
}

access_set!(ReadSet);
access_set!(WriteSet);

/// Return whether a canonical read set and write set address any common state.
///
/// This is the shared implementation for every overlap-driven feature. The
/// walk is linear in the two normalized set sizes.
pub fn read_write_overlap(reads: &ReadSet, writes: &WriteSet) -> bool {
    overlap_walk(reads.entries(), writes.entries())
}

fn normalize(paths: impl IntoIterator<Item = AccessPath>) -> Result<Vec<AccessPath>, OverlapError> {
    let mut paths = paths
        .into_iter()
        .map(normalize_path)
        .collect::<Result<Vec<_>, _>>()?;
    if paths
        .iter()
        .any(|path| matches!(path, AccessPath::Wildcard))
    {
        return Ok(vec![AccessPath::Wildcard]);
    }

    paths.sort_by(compare_paths);
    paths.dedup();

    let entity_scopes = paths
        .iter()
        .filter_map(|path| match path {
            AccessPath::Entity { entity } => Some(entity.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let attribute_scopes = paths
        .iter()
        .filter_map(|path| match path {
            AccessPath::Attribute { entity, attribute } => {
                Some((entity.clone(), attribute.clone()))
            }
            _ => None,
        })
        .collect::<Vec<_>>();

    paths.retain(|path| match path {
        AccessPath::Wildcard | AccessPath::Entity { .. } => true,
        AccessPath::Attribute { entity, .. } => entity_scopes.binary_search(entity).is_err(),
        AccessPath::Interval {
            entity, attribute, ..
        } => {
            entity_scopes.binary_search(entity).is_err()
                && attribute_scopes
                    .binary_search(&(entity.clone(), attribute.clone()))
                    .is_err()
        }
    });

    Ok(merge_intervals(paths))
}

fn normalize_path(path: AccessPath) -> Result<AccessPath, OverlapError> {
    Ok(match path {
        AccessPath::Wildcard => AccessPath::Wildcard,
        AccessPath::Entity { entity } => AccessPath::Entity {
            entity: identifier(entity, "entity")?,
        },
        AccessPath::Attribute { entity, attribute } => AccessPath::Attribute {
            entity: identifier(entity, "entity")?,
            attribute: identifier(attribute, "attribute")?,
        },
        AccessPath::Interval {
            entity,
            attribute,
            start,
            end,
        } => {
            if start >= end {
                return Err(OverlapError::EmptyInterval { start, end });
            }
            AccessPath::Interval {
                entity: identifier(entity, "entity")?,
                attribute: identifier(attribute, "attribute")?,
                start,
                end,
            }
        }
    })
}

fn identifier(value: String, field: &'static str) -> Result<String, OverlapError> {
    let value = value.trim();
    if value.is_empty() {
        Err(OverlapError::EmptyIdentifier(field))
    } else {
        Ok(value.to_string())
    }
}

fn compare_paths(left: &AccessPath, right: &AccessPath) -> Ordering {
    path_key(left).cmp(&path_key(right))
}

fn path_key(path: &AccessPath) -> (&str, &str, u8, i64, i64) {
    match path {
        AccessPath::Wildcard => ("", "", 0, 0, 0),
        AccessPath::Entity { entity } => (entity, "", 0, 0, 0),
        AccessPath::Attribute { entity, attribute } => (entity, attribute, 1, 0, 0),
        AccessPath::Interval {
            entity,
            attribute,
            start,
            end,
        } => (entity, attribute, 2, *start, *end),
    }
}

fn merge_intervals(paths: Vec<AccessPath>) -> Vec<AccessPath> {
    let mut merged = Vec::with_capacity(paths.len());
    for path in paths {
        if let AccessPath::Interval {
            entity,
            attribute,
            start,
            end,
        } = &path
        {
            if let Some(AccessPath::Interval {
                entity: previous_entity,
                attribute: previous_attribute,
                end: previous_end,
                ..
            }) = merged.last_mut()
            {
                if previous_entity == entity
                    && previous_attribute == attribute
                    && start <= previous_end
                {
                    *previous_end = (*previous_end).max(*end);
                    continue;
                }
            }
        }
        merged.push(path);
    }
    merged
}

fn overlap_walk(left: &[AccessPath], right: &[AccessPath]) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }
    if matches!(left.first(), Some(AccessPath::Wildcard))
        || matches!(right.first(), Some(AccessPath::Wildcard))
    {
        return true;
    }

    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        let left_entity = entity_of(&left[left_index]);
        let right_entity = entity_of(&right[right_index]);
        match left_entity.cmp(right_entity) {
            Ordering::Less => left_index = end_of_entity(left, left_index),
            Ordering::Greater => right_index = end_of_entity(right, right_index),
            Ordering::Equal => {
                let left_end = end_of_entity(left, left_index);
                let right_end = end_of_entity(right, right_index);
                if entity_group_overlaps(
                    &left[left_index..left_end],
                    &right[right_index..right_end],
                ) {
                    return true;
                }
                left_index = left_end;
                right_index = right_end;
            }
        }
    }
    false
}

fn entity_of(path: &AccessPath) -> &str {
    match path {
        AccessPath::Wildcard => "",
        AccessPath::Entity { entity }
        | AccessPath::Attribute { entity, .. }
        | AccessPath::Interval { entity, .. } => entity,
    }
}

fn end_of_entity(paths: &[AccessPath], start: usize) -> usize {
    let entity = entity_of(&paths[start]);
    paths[start..]
        .iter()
        .position(|path| entity_of(path) != entity)
        .map_or(paths.len(), |offset| start + offset)
}

fn entity_group_overlaps(left: &[AccessPath], right: &[AccessPath]) -> bool {
    if matches!(left.first(), Some(AccessPath::Entity { .. }))
        || matches!(right.first(), Some(AccessPath::Entity { .. }))
    {
        return true;
    }

    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        let left_attribute = attribute_of(&left[left_index]);
        let right_attribute = attribute_of(&right[right_index]);
        match left_attribute.cmp(right_attribute) {
            Ordering::Less => left_index = end_of_attribute(left, left_index),
            Ordering::Greater => right_index = end_of_attribute(right, right_index),
            Ordering::Equal => {
                let left_end = end_of_attribute(left, left_index);
                let right_end = end_of_attribute(right, right_index);
                if attribute_group_overlaps(
                    &left[left_index..left_end],
                    &right[right_index..right_end],
                ) {
                    return true;
                }
                left_index = left_end;
                right_index = right_end;
            }
        }
    }
    false
}

fn attribute_of(path: &AccessPath) -> &str {
    match path {
        AccessPath::Attribute { attribute, .. } | AccessPath::Interval { attribute, .. } => {
            attribute
        }
        AccessPath::Wildcard | AccessPath::Entity { .. } => "",
    }
}

fn end_of_attribute(paths: &[AccessPath], start: usize) -> usize {
    let attribute = attribute_of(&paths[start]);
    paths[start..]
        .iter()
        .position(|path| attribute_of(path) != attribute)
        .map_or(paths.len(), |offset| start + offset)
}

fn attribute_group_overlaps(left: &[AccessPath], right: &[AccessPath]) -> bool {
    if matches!(left.first(), Some(AccessPath::Attribute { .. }))
        || matches!(right.first(), Some(AccessPath::Attribute { .. }))
    {
        return true;
    }

    let (mut left_index, mut right_index) = (0, 0);
    while left_index < left.len() && right_index < right.len() {
        let (left_start, left_end) = interval_of(&left[left_index]);
        let (right_start, right_end) = interval_of(&right[right_index]);
        if left_start < right_end && right_start < left_end {
            return true;
        }
        if left_end <= right_end {
            left_index += 1;
        } else {
            right_index += 1;
        }
    }
    false
}

fn interval_of(path: &AccessPath) -> (i64, i64) {
    match path {
        AccessPath::Interval { start, end, .. } => (*start, *end),
        AccessPath::Wildcard | AccessPath::Entity { .. } | AccessPath::Attribute { .. } => {
            unreachable!("only interval groups reach the interval overlap walk")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_removes_subsumed_paths_and_merges_ranges() {
        let set = ReadSet::new([
            AccessPath::Interval {
                entity: " note:1 ".to_string(),
                attribute: " body ".to_string(),
                start: 10,
                end: 20,
            },
            AccessPath::Interval {
                entity: "note:1".to_string(),
                attribute: "body".to_string(),
                start: 0,
                end: 10,
            },
            AccessPath::Attribute {
                entity: "note:1".to_string(),
                attribute: "body".to_string(),
            },
        ])
        .unwrap();

        assert_eq!(
            set.entries(),
            &[AccessPath::Attribute {
                entity: "note:1".to_string(),
                attribute: "body".to_string(),
            }]
        );
    }

    #[test]
    fn invalid_identifiers_and_empty_intervals_are_rejected() {
        assert_eq!(
            ReadSet::new([AccessPath::Entity {
                entity: "  ".to_string(),
            }]),
            Err(OverlapError::EmptyIdentifier("entity"))
        );
        assert_eq!(
            WriteSet::new([AccessPath::Interval {
                entity: "note:1".to_string(),
                attribute: "body".to_string(),
                start: 4,
                end: 4,
            }]),
            Err(OverlapError::EmptyInterval { start: 4, end: 4 })
        );
    }
}
