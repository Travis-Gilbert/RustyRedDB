use std::collections::BTreeMap;

use rustyred_thg_core::{GraphStoreError, NeighborHit, NeighborQuery, NodeQuery, NodeRecord};
use rustyred_thg_query::{
    classify_ast, execute_read, parse_cypher_pest, write_clause_name, ClauseClass, CypherGraph,
    PlanSteeringState,
};
use serde::Serialize;
use serde_json::Value;

use crate::error::{Error, ErrorKind};
use crate::Database;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
    pub truncated: bool,
}

struct Adapter<'a> {
    graph: &'a rustyred_thg_core::RedCoreGraphStore,
}

impl CypherGraph for Adapter<'_> {
    fn get_node(&self, id: &str) -> Result<Option<NodeRecord>, GraphStoreError> {
        self.graph.get_node(id)
    }

    fn query_nodes(&self, query: NodeQuery) -> Result<Vec<NodeRecord>, GraphStoreError> {
        self.graph.query_nodes(query)
    }

    fn neighbors(&self, query: NeighborQuery) -> Result<Vec<NeighborHit>, GraphStoreError> {
        self.graph.neighbors(query)
    }

    fn graph_version(&self) -> Result<u64, GraphStoreError> {
        Ok(self.graph.stats()?.version)
    }
}

impl Database {
    /// Run one read-only Cypher query. Parameters are bound values, not text
    /// spliced into the query. Write clauses are refused here; use [`Database::mutate`].
    pub fn query(
        &self,
        cypher: &str,
        params: &BTreeMap<String, Value>,
    ) -> Result<QueryResult, Error> {
        self.query_with_limit(cypher, params, 1_000)
    }

    pub fn query_with_limit(
        &self,
        cypher: &str,
        params: &BTreeMap<String, Value>,
        row_cap: usize,
    ) -> Result<QueryResult, Error> {
        let parsed = parse_cypher_pest(cypher, params).map_err(Error::from_query)?;
        match classify_ast(&parsed).map_err(Error::from_query)? {
            ClauseClass::Read => {}
            ClauseClass::Write => {
                let clause = write_clause_name(&parsed).unwrap_or("WRITE");
                return Err(Error::new(
                    ErrorKind::Query,
                    "cypher_write_unsupported",
                    format!(
                        "{clause} is not executed by the read path; commit a GraphMutationBatch instead"
                    ),
                ));
            }
        }
        let outcome = execute_read(
            &Adapter {
                graph: self.engine(),
            },
            &parsed,
            None::<&PlanSteeringState>,
            row_cap,
        )
        .map_err(Error::from_query)?;
        Ok(QueryResult {
            columns: outcome.columns,
            rows: outcome.rows,
            truncated: outcome.truncated,
        })
    }
}
