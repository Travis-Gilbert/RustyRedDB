use std::fmt;

use rustyred_thg_core::GraphStoreError;
use rustyred_thg_query::CypherError;
use serde::Serialize;

/// Why a database call stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    /// `create` found data already at the path.
    AlreadyExists,
    /// `open` was asked for a database that is not there.
    NotFound,
    /// A write was attempted through a read-only handle.
    ReadOnly,
    /// The Cypher text or its parameters were rejected.
    Query,
    /// The engine refused before publication, or the file could not be read.
    Storage,
}

/// A bounded failure. `code` is stable for programs. The message is for people.
///
/// A successful [`crate::Database::mutate`] means the engine published the
/// commit. This type is not returned for a hook or pin-cleanup problem that
/// happens after that publication: those stay on the engine's `Ok` path so a
/// caller does not retry a write that already landed.
#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    pub kind: ErrorKind,
    pub code: String,
    pub message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind,
            code: code.into(),
            message: message.into(),
        }
    }

    pub(crate) fn from_storage(error: GraphStoreError) -> Self {
        let kind = match error.code.as_str() {
            "rr_already_exists" => ErrorKind::AlreadyExists,
            "not_found" => ErrorKind::NotFound,
            _ => ErrorKind::Storage,
        };
        Self {
            kind,
            code: error.code,
            message: error.message,
        }
    }

    pub(crate) fn from_query(error: CypherError) -> Self {
        Self {
            kind: ErrorKind::Query,
            code: error.code,
            message: error.message,
        }
    }

    pub fn body(&self) -> ErrorBody<'_> {
        ErrorBody {
            ok: false,
            code: &self.code,
            message: &self.message,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

/// Machine-readable failure written to stdout by the CLI.
#[derive(Debug, Serialize)]
pub struct ErrorBody<'a> {
    pub ok: bool,
    pub code: &'a str,
    pub message: &'a str,
}
