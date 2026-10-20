use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum WikiRefusal {
    AuthenticationRequired,
    ProjectRequired,
    ScopeRequired,
    Excluded,
    Missing { target: String },
    AmbiguousIdentity,
    SummaryRequired,
    RevisionConflict,
    InvalidRequest { message: String },
    Incomplete { message: String },
    Unavailable { message: String },
}
