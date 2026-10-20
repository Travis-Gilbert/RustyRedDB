//! Embedded Rusty Red.
//!
//! [`Database`] is the supported entry point. It creates a native `.RR` file,
//! runs read-only Cypher, and commits typed graph mutations through the same
//! engine Theorem uses. Lower-level modules stay available on
//! [`rustyred_thg_core`] and [`rustyred_thg_query`] for callers that already
//! know those types.
//!
//! Opening a database does not create a project, choose an embedding model,
//! or contact the network. Vector values are whatever the caller stores.

mod database;
mod error;
mod options;
mod query;

pub use database::{Database, Inspection, MigrationReport};
pub use error::{Error, ErrorBody, ErrorKind};
pub use options::OpenOptions;
pub use query::QueryResult;

pub use rustyred_thg_core::{
    EdgeRecord, GraphMutation, GraphMutationBatch, GraphStats, GraphStoreError, GraphTransaction,
    NodeRecord, RedCoreDurability, RedCoreGraphStore, RedCoreOptions, VerifyReport,
};
pub use rustyred_thg_query::{execute_read, parse_cypher_pest, CypherError, CypherGraph};
