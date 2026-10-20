//! Shared OpenCypher parser, planner, executor, and query receipts.
//!
//! One pest grammar is the gate for HTTP (`rustyred-thg-server`) and MCP
//! (`rustyred-thg-mcp`). MCP cannot depend on the server crate.

pub mod ast_hash;
pub mod cypher;
pub mod dialect;
pub mod error;
pub mod execute;
pub mod read_gate;
pub mod receipt;

pub use ast_hash::{ast_hash, params_hash, sql_ast_hash};
pub use cypher::planner::{
    PlanSteerSlot, PlanSteeringState, PlannerSteeringDecision, PLAN_CANDIDATE_EXPAND_LEFT_OUT,
    PLAN_CANDIDATE_EXPAND_RIGHT_IN,
};
pub use cypher::{parse_cypher_pest, CypherPestParser, Rule};
pub use dialect::{
    detect_sql_unsupported, detect_unsupported, dialect_block, parse_probe, supported_alternative,
    DialectBlock, SUPPORTED_PROBES, UNSUPPORTED_CONSTRUCTS, VARIABLE_LENGTH_PROBES,
};
pub use error::{json_span, CypherError, QuerySpan};
pub use execute::{execute_read, explain_read, CypherGraph, PlanPreview, ReadOutcome};
pub use read_gate::{
    classify_ast, referenced_edge_types, referenced_labels, referenced_properties,
    write_clause_name, ClauseClass,
};
pub use receipt::{MemoEntry, MemoKey, QueryReceipt};
