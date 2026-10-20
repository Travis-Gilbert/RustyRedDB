use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::access_method::{ColumnId, RelationId, RowId, ScalarValue};

/// An ephemeral relational read result resolved from a live graph record.
///
/// `ResolvedRow` may own cloned values for the duration of a query, but no
/// [`crate::RelationalLens`] stores one. The graph record remains the only
/// authoritative home for every value.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ResolvedRow {
    pub id: RowId,
    pub relation: RelationId,
    pub values: BTreeMap<ColumnId, ScalarValue>,
    #[serde(default)]
    pub properties: Value,
}

impl ResolvedRow {
    pub fn new(
        relation: impl Into<String>,
        id: impl Into<String>,
        values: BTreeMap<ColumnId, ScalarValue>,
    ) -> Self {
        Self {
            id: id.into(),
            relation: relation.into(),
            values,
            properties: json!({}),
        }
    }

    pub fn value(&self, column: &str) -> Option<&ScalarValue> {
        if column == "id" {
            return None;
        }
        self.values.get(column)
    }
}
