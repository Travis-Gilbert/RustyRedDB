//! Browser binding for embedded Rusty Red.
//!
//! SOURCING: wasm-bindgen generates the JavaScript glue. The storage is the
//! engine's own browser projection (`rustyred_thg_core::browser_projection`)
//! over an OPFS synchronous access handle; this crate adds no storage code.
//!
//! The surface is the Node binding's, reached through one entry point:
//! [`Session::call`] takes a method name and a JSON value and returns a JSON
//! value. One entry keeps the wasm boundary to strings, and lets a native
//! `cargo test` exercise the same table the browser uses. The worker and the
//! asynchronous client that people import are JavaScript, in `npm/`.
//!
//! A failure carries the facade's stable `code`, such as
//! `cypher_write_unsupported`, as the Node binding's do.

use std::collections::BTreeMap;

use rustyred::engine::{NeighborQuery, NodeQuery};
use rustyred::{EdgeRecord, GraphMutation, GraphMutationBatch, NodeRecord};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

/// Why a call stopped. `code` is stable for programs.
#[derive(Clone, Debug, PartialEq)]
pub struct Failure {
    pub code: String,
    pub message: String,
}

impl Failure {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
        }
    }
}

impl From<rustyred::Error> for Failure {
    fn from(error: rustyred::Error) -> Self {
        Self {
            code: error.code,
            message: error.message,
        }
    }
}

impl From<rustyred::GraphStoreError> for Failure {
    fn from(error: rustyred::GraphStoreError) -> Self {
        Self {
            code: error.code,
            message: error.message,
        }
    }
}

type Outcome<T> = Result<T, Failure>;

/// How to open a database. Both choices belong to this handle and are not stored.
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OpenOptions {
    /// A write to a labelled node also commits a `ControlEvent` node recording
    /// it. Defaults to true; turn it off for derived data such as an index.
    pub control_events: Option<bool>,
    /// Each commit joins the recorded revision history. Defaults to true.
    pub keep_history: Option<bool>,
}

#[cfg(target_arch = "wasm32")]
impl OpenOptions {
    fn facade(&self) -> rustyred::OpenOptions {
        let mut open = rustyred::OpenOptions::new();
        if let Some(enabled) = self.control_events {
            open = open.control_events(enabled);
        }
        if let Some(keep) = self.keep_history {
            open = open.keep_history(keep);
        }
        open
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Node {
    id: String,
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default = "empty_object")]
    properties: Value,
}

impl From<NodeRecord> for Node {
    fn from(node: NodeRecord) -> Self {
        Self {
            id: node.id,
            labels: node.labels,
            properties: node.properties,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Edge {
    id: String,
    from: String,
    to: String,
    #[serde(rename = "type")]
    edge_type: String,
    #[serde(default = "empty_object")]
    properties: Value,
}

impl From<EdgeRecord> for Edge {
    fn from(edge: EdgeRecord) -> Self {
        Self {
            id: edge.id,
            from: edge.from_id,
            to: edge.to_id,
            edge_type: edge.edge_type,
            properties: edge.properties,
        }
    }
}

fn empty_object() -> Value {
    Value::Object(Map::new())
}

/// One atomic commit. Deletes apply before upserts, whatever the order of the
/// lists. Deleting a node also deletes its edges, except those this batch
/// upserts. An upsert replaces the whole record, and an edge's ends must exist
/// once the batch's node upserts are applied.
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Batch {
    #[serde(default)]
    delete_edges: Vec<String>,
    #[serde(default)]
    delete_nodes: Vec<String>,
    #[serde(default)]
    upsert_nodes: Vec<Node>,
    #[serde(default)]
    upsert_edges: Vec<Edge>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PropertyPrefix {
    property: String,
    value: String,
}

/// Which nodes `queryNodes` returns. Every given condition must hold.
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeFilter {
    label: Option<String>,
    properties: Option<Map<String, Value>>,
    prefix: Option<PropertyPrefix>,
    limit: Option<u32>,
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Direction {
    Out,
    In,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NeighborFilter {
    node_id: String,
    direction: Direction,
    edge_type: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct QueryCall {
    cypher: String,
    #[serde(default)]
    params: BTreeMap<String, Value>,
    row_cap: Option<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct IdCall {
    id: String,
}

fn arguments<T: for<'de> Deserialize<'de>>(method: &str, args: Value) -> Outcome<T> {
    serde_json::from_value(args)
        .map_err(|error| Failure::new("invalid_arguments", format!("{method}: {error}")))
}

/// One open database and the calls a host may make on it.
pub struct Session {
    database: Option<rustyred::Database>,
}

impl Session {
    /// An empty graph with no file and no durability.
    pub fn in_memory() -> Self {
        Self::over(rustyred::Database::in_memory())
    }

    pub fn over(database: rustyred::Database) -> Self {
        Self {
            database: Some(database),
        }
    }

    fn handle(&self) -> Outcome<&rustyred::Database> {
        self.database
            .as_ref()
            .ok_or_else(|| Failure::new("database_closed", "the database is closed"))
    }

    fn handle_mut(&mut self) -> Outcome<&mut rustyred::Database> {
        self.database
            .as_mut()
            .ok_or_else(|| Failure::new("database_closed", "the database is closed"))
    }

    fn live_node(&self, id: &str) -> Outcome<Option<NodeRecord>> {
        let node = self.handle()?.engine().get_node(id)?;
        Ok(node.filter(|node| !node.tombstone))
    }

    /// Run one method. The names and shapes are `@rustyreddb/node`'s.
    pub fn call(&mut self, method: &str, args: Value) -> Outcome<Value> {
        match method {
            "close" => {
                self.database = None;
                Ok(Value::Null)
            }
            "inspect" => {
                let inspection = self.handle()?.inspect()?;
                Ok(json!({
                    "writable": inspection.writable,
                    "version": inspection.version,
                    "nodes": inspection.nodes,
                    "edges": inspection.edges,
                }))
            }
            "verify" => {
                let report = self.handle()?.verify()?;
                serde_json::to_value(report).map_err(|error| {
                    Failure::new("verify_report_unserializable", error.to_string())
                })
            }
            "mutate" => self.mutate(arguments(method, args)?),
            "getNode" => {
                let call: IdCall = arguments(method, args)?;
                Ok(json!(self.live_node(&call.id)?.map(Node::from)))
            }
            "getEdge" => {
                let call: IdCall = arguments(method, args)?;
                let edge = self.handle()?.engine().get_edge(&call.id)?;
                Ok(json!(edge.filter(|edge| !edge.tombstone).map(Edge::from)))
            }
            "queryNodes" => self.query_nodes(arguments(method, args)?),
            "neighbors" => self.neighbors(arguments(method, args)?),
            "query" => {
                let call: QueryCall = arguments(method, args)?;
                let result = self.handle()?.query_with_limit(
                    &call.cypher,
                    &call.params,
                    call.row_cap.unwrap_or(1_000) as usize,
                )?;
                Ok(json!({
                    "columns": result.columns,
                    "rows": result.rows,
                    "truncated": result.truncated,
                }))
            }
            // Full-text search is the Node binding's and not yet this one's: the
            // engine's index is built with Tantivy, whose default features do
            // not build for wasm32-unknown-unknown.
            "designateFulltext" | "fulltextSearch" => Err(Failure::new(
                "fulltext_unavailable",
                "full-text search is not available in the browser binding yet",
            )),
            other => Err(Failure::new(
                "unknown_method",
                format!("{other} is not a database method"),
            )),
        }
    }

    fn mutate(&mut self, batch: Batch) -> Outcome<Value> {
        let mutations = batch
            .delete_edges
            .into_iter()
            .map(GraphMutation::EdgeDelete)
            .chain(
                batch
                    .delete_nodes
                    .into_iter()
                    .map(GraphMutation::NodeDelete),
            )
            .chain(batch.upsert_nodes.into_iter().map(|node| {
                GraphMutation::NodeUpsert(NodeRecord::new(node.id, node.labels, node.properties))
            }))
            .chain(batch.upsert_edges.into_iter().map(|edge| {
                GraphMutation::EdgeUpsert(EdgeRecord::new(
                    edge.id,
                    edge.from,
                    edge.edge_type,
                    edge.to,
                    edge.properties,
                ))
            }));
        let transaction = self
            .handle_mut()?
            .mutate(GraphMutationBatch::new(mutations))?;
        Ok(json!({
            "txnId": transaction.txn_id,
            "graphVersion": transaction.graph_version,
        }))
    }

    fn query_nodes(&self, filter: NodeFilter) -> Outcome<Value> {
        let properties: BTreeMap<String, Value> =
            filter.properties.unwrap_or_default().into_iter().collect();
        let limit = filter.limit.map(|limit| limit as usize);
        let engine = self.handle()?.engine();
        let Some(prefix) = filter.prefix else {
            let nodes = engine.query_nodes(NodeQuery {
                label: filter.label,
                properties,
                limit,
                include_expired: false,
            })?;
            return Ok(json!(nodes.into_iter().map(Node::from).collect::<Vec<_>>()));
        };
        let ids = match &filter.label {
            Some(label) => engine.node_ids_for_label_and_string_property_prefix(
                label,
                &prefix.property,
                &prefix.value,
            ),
            None => engine.node_ids_for_string_property_prefix(&prefix.property, &prefix.value),
        }?;
        let mut nodes = Vec::new();
        for id in ids {
            if limit.is_some_and(|limit| nodes.len() >= limit) {
                break;
            }
            let Some(node) = self.live_node(&id)? else {
                continue;
            };
            let matches = properties
                .iter()
                .all(|(key, value)| node.properties.get(key) == Some(value));
            if matches {
                nodes.push(Node::from(node));
            }
        }
        Ok(json!(nodes))
    }

    fn neighbors(&self, filter: NeighborFilter) -> Outcome<Value> {
        let query = match filter.direction {
            Direction::Out => NeighborQuery::out(filter.node_id),
            Direction::In => NeighborQuery::in_(filter.node_id),
        };
        let query = match filter.edge_type {
            Some(edge_type) => query.with_edge_type(edge_type),
            None => query,
        };
        let engine = self.handle()?.engine();
        let mut neighbors = Vec::new();
        for hit in engine.neighbors(query)? {
            let Some(edge) = engine.get_edge(&hit.edge_id)? else {
                continue;
            };
            if edge.tombstone {
                continue;
            }
            neighbors.push(json!({
                "edgeId": hit.edge_id,
                "nodeId": hit.node_id,
                "type": hit.edge_type,
                "properties": edge.properties,
            }));
        }
        Ok(Value::Array(neighbors))
    }
}

#[cfg(target_arch = "wasm32")]
mod glue {
    use std::sync::Arc;

    use rustyred::engine::browser_projection::ProjectionChunkStore;
    use rustyred::engine::UnifiedChunkStore;
    use wasm_bindgen::prelude::*;

    use super::{Failure, OpenOptions, Session};

    fn thrown(failure: Failure) -> JsValue {
        let error = js_sys::Error::new(&failure.message);
        let _ = js_sys::Reflect::set(&error, &"code".into(), &failure.code.into());
        error.into()
    }

    /// A database held by the worker that opened it.
    #[wasm_bindgen]
    pub struct Database {
        session: Session,
    }

    #[wasm_bindgen]
    impl Database {
        /// Open, creating when absent, the database in the OPFS directory
        /// `directory`. Only a dedicated worker can do this, and one worker
        /// holds the directory at a time: the browser refuses a second.
        pub async fn open(directory: String, options: Option<String>) -> Result<Database, JsValue> {
            let options: OpenOptions = match options {
                None => OpenOptions::default(),
                Some(text) => serde_json::from_str(&text).map_err(|error| {
                    thrown(Failure::new("invalid_arguments", format!("open: {error}")))
                })?,
            };
            let store = ProjectionChunkStore::open_in_worker(&directory)
                .await
                .map_err(|error| thrown(error.into()))?;
            let store: Arc<dyn UnifiedChunkStore> = Arc::new(store);
            let database = rustyred::Database::with_chunk_store(store, &options.facade())
                .map_err(|error| thrown(error.into()))?;
            Ok(Database {
                session: Session::over(database),
            })
        }

        /// An empty graph with no file and no durability.
        #[wasm_bindgen(js_name = inMemory)]
        pub fn in_memory() -> Database {
            Database {
                session: Session::in_memory(),
            }
        }

        /// Run one method with JSON arguments and return JSON.
        pub fn call(&mut self, method: &str, args: &str) -> Result<String, JsValue> {
            let args = serde_json::from_str(args).map_err(|error| {
                thrown(Failure::new(
                    "invalid_arguments",
                    format!("{method}: {error}"),
                ))
            })?;
            let value = self.session.call(method, args).map_err(thrown)?;
            Ok(value.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded() -> Session {
        let mut session = Session::in_memory();
        session
            .call(
                "mutate",
                json!({
                    "upsertNodes": [
                        { "id": "org:a", "labels": ["Org"], "properties": { "name": "Alder", "area": "housing" } },
                        { "id": "org:b", "labels": ["Org"], "properties": { "name": "Birch", "area": "transit" } },
                    ],
                    "upsertEdges": [
                        { "id": "e:1", "from": "org:a", "to": "org:b", "type": "FUNDS", "properties": { "year": 2026 } },
                    ],
                }),
            )
            .expect("seed");
        session
    }

    #[test]
    fn a_batch_commits_and_reads_back_in_the_node_bindings_shapes() {
        let mut session = seeded();
        let node = session.call("getNode", json!({ "id": "org:a" })).unwrap();
        assert_eq!(node["labels"], json!(["Org"]));
        assert_eq!(node["properties"]["name"], "Alder");

        let edge = session.call("getEdge", json!({ "id": "e:1" })).unwrap();
        assert_eq!(edge["from"], "org:a");
        assert_eq!(edge["type"], "FUNDS");

        let housing = session
            .call(
                "queryNodes",
                json!({ "label": "Org", "properties": { "area": "housing" } }),
            )
            .unwrap();
        assert_eq!(housing.as_array().unwrap().len(), 1);

        let out = session
            .call(
                "neighbors",
                json!({ "nodeId": "org:a", "direction": "out" }),
            )
            .unwrap();
        assert_eq!(out[0]["nodeId"], "org:b");
        assert_eq!(out[0]["properties"]["year"], 2026);
    }

    #[test]
    fn cypher_reads_with_bound_parameters_and_refuses_writes() {
        let mut session = seeded();
        let result = session
            .call(
                "query",
                json!({
                    "cypher": "MATCH (o:Org) WHERE o.area = $area RETURN o.name AS name",
                    "params": { "area": "transit" },
                }),
            )
            .unwrap();
        assert_eq!(result["columns"], json!(["name"]));
        assert_eq!(result["rows"], json!([["Birch"]]));

        let refused = session
            .call("query", json!({ "cypher": "CREATE (o:Org {name: 'x'})" }))
            .unwrap_err();
        assert_eq!(refused.code, "cypher_write_unsupported");
    }

    #[test]
    fn a_delete_takes_the_node_and_its_edges() {
        let mut session = seeded();
        session
            .call("mutate", json!({ "deleteNodes": ["org:b"] }))
            .unwrap();
        assert_eq!(
            session.call("getNode", json!({ "id": "org:b" })).unwrap(),
            Value::Null
        );
        assert_eq!(
            session.call("getEdge", json!({ "id": "e:1" })).unwrap(),
            Value::Null
        );
    }

    #[test]
    fn mistakes_are_named() {
        let mut session = seeded();
        let unknown = session.call("dropEverything", Value::Null).unwrap_err();
        assert_eq!(unknown.code, "unknown_method");

        let misspelt = session
            .call("mutate", json!({ "upsertNode": [] }))
            .unwrap_err();
        assert_eq!(misspelt.code, "invalid_arguments");

        session.call("close", Value::Null).unwrap();
        let closed = session.call("inspect", Value::Null).unwrap_err();
        assert_eq!(closed.code, "database_closed");
    }
}
