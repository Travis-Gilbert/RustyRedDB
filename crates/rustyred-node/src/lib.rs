//! Node-API binding for embedded Rusty Red, for Node.js and Bun.
//!
//! SOURCING: napi-rs (`napi`, `napi-derive` and `@napi-rs/cli`) generates the
//! Node-API glue, the TypeScript declarations and the platform loader. This
//! file only maps the [`rustyred::Database`] facade onto JavaScript values.
//!
//! Calls are synchronous and run on the JavaScript thread, as the facade's are.
//! A failure throws an `Error` whose `code` is the facade's stable error code,
//! such as `rr_already_exists` or `fulltext_not_designated`.

use std::collections::{BTreeMap, HashMap};

use napi_derive::napi;
use rustyred::engine::{NeighborQuery, NodeQuery};
use rustyred::{EdgeRecord, GraphMutation, GraphMutationBatch, NodeRecord};
use serde_json::Value;

type Result<T> = napi::Result<T, String>;

fn failure(error: rustyred::Error) -> napi::Error<String> {
    napi::Error::new(error.code, error.message)
}

fn storage(error: rustyred::GraphStoreError) -> napi::Error<String> {
    napi::Error::new(error.code, error.message)
}

/// How to create or open a database. Every choice belongs to this handle and
/// is not stored in the file.
#[napi(object)]
#[derive(Default)]
pub struct OpenOptions {
    /// The writer identity recorded in the file's claim. Defaults to `local`.
    pub device_name: Option<String>,
    /// A write to a labelled node also commits a `ControlEvent` node recording
    /// it. Defaults to true; turn it off for derived data such as an index.
    pub control_events: Option<bool>,
    /// Each commit joins the recorded revision history. Defaults to true. Off,
    /// the file keeps only its current state and `compact()` reclaims what each
    /// write replaced.
    pub keep_history: Option<bool>,
}

fn open_options(options: Option<OpenOptions>) -> rustyred::OpenOptions {
    let options = options.unwrap_or_default();
    let mut open = rustyred::OpenOptions::new();
    if let Some(name) = options.device_name {
        open = open.device_name(name);
    }
    if let Some(enabled) = options.control_events {
        open = open.control_events(enabled);
    }
    if let Some(keep) = options.keep_history {
        open = open.keep_history(keep);
    }
    open
}

#[napi(object)]
pub struct Node {
    pub id: String,
    pub labels: Vec<String>,
    #[napi(ts_type = "Record<string, unknown>")]
    pub properties: Value,
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

#[napi(object)]
pub struct Edge {
    pub id: String,
    pub from: String,
    pub to: String,
    #[napi(js_name = "type")]
    pub edge_type: String,
    #[napi(ts_type = "Record<string, unknown>")]
    pub properties: Value,
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

/// One atomic commit. Deletes apply before upserts, whatever the order of the
/// lists. Deleting a node also deletes its edges, except those this batch
/// upserts. An upsert replaces the whole record, and an edge's ends must exist
/// once the batch's node upserts are applied.
#[napi(object)]
pub struct Batch {
    pub delete_edges: Option<Vec<String>>,
    pub delete_nodes: Option<Vec<String>>,
    pub upsert_nodes: Option<Vec<Node>>,
    pub upsert_edges: Option<Vec<Edge>>,
}

#[napi(object)]
pub struct Commit {
    pub txn_id: i64,
    pub graph_version: i64,
}

#[napi(object)]
pub struct PropertyPrefix {
    pub property: String,
    pub value: String,
}

/// Which nodes `queryNodes` returns. Every given condition must hold.
#[napi(object)]
pub struct NodeFilter {
    pub label: Option<String>,
    /// Properties each node must equal. Strings, numbers, booleans and null
    /// are indexed; arrays and objects are not.
    #[napi(ts_type = "Record<string, unknown>")]
    pub properties: Option<Value>,
    /// A string property each node's value must start with. The nodes then
    /// come in byte order of that value.
    pub prefix: Option<PropertyPrefix>,
    pub limit: Option<u32>,
}

#[napi(string_enum = "lowercase")]
pub enum Direction {
    Out,
    In,
}

#[napi(object)]
pub struct NeighborFilter {
    pub node_id: String,
    pub direction: Direction,
    pub edge_type: Option<String>,
}

/// A node next to the one asked about, and the edge between them.
#[napi(object)]
pub struct Neighbor {
    pub edge_id: String,
    pub node_id: String,
    #[napi(js_name = "type")]
    pub edge_type: String,
    /// The edge's properties.
    #[napi(ts_type = "Record<string, unknown>")]
    pub properties: Value,
}

#[napi(object)]
pub struct QueryResult {
    pub columns: Vec<String>,
    #[napi(ts_type = "Array<Array<unknown>>")]
    pub rows: Vec<Vec<Value>>,
    pub truncated: bool,
}

#[napi(object)]
pub struct FullTextHit {
    pub id: String,
    pub score: f64,
    /// HTML-escaped text around the matches, each match in `<mark>`.
    pub snippet: String,
}

/// What `compact()` removed. Bytes count chunk contents, without the file's
/// own framing, so the file is somewhat larger than `bytesAfter`.
#[napi(object)]
pub struct CompactionReport {
    pub chunks_before: i64,
    pub chunks_after: i64,
    pub chunks_removed: i64,
    pub bytes_before: i64,
    pub bytes_after: i64,
    pub bytes_removed: i64,
}

#[napi(object)]
pub struct Inspection {
    pub path: Option<String>,
    pub database_id: Option<String>,
    pub writable: bool,
    pub version: i64,
    pub nodes: i64,
    pub edges: i64,
}

/// An embedded Rusty Red database: a native `.rr` file, or a graph in memory.
#[napi]
pub struct Database {
    inner: Option<rustyred::Database>,
}

impl Database {
    fn handle(&self) -> Result<&rustyred::Database> {
        self.inner
            .as_ref()
            .ok_or_else(|| napi::Error::new("database_closed".into(), "the database is closed"))
    }

    fn handle_mut(&mut self) -> Result<&mut rustyred::Database> {
        self.inner
            .as_mut()
            .ok_or_else(|| napi::Error::new("database_closed".into(), "the database is closed"))
    }

    fn live_node(&self, id: &str) -> Result<Option<NodeRecord>> {
        let node = self.handle()?.engine().get_node(id).map_err(storage)?;
        Ok(node.filter(|node| !node.tombstone))
    }
}

#[napi]
impl Database {
    /// Create a new `.rr` file. Refuses a path that already exists.
    #[napi(factory)]
    pub fn create(path: String, options: Option<OpenOptions>) -> Result<Self> {
        let inner =
            rustyred::Database::create_with(path, &open_options(options)).map_err(failure)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Open an existing `.rr` file for writing. A second writer is refused
    /// while this handle is open.
    #[napi(factory)]
    pub fn open(path: String, options: Option<OpenOptions>) -> Result<Self> {
        let inner = rustyred::Database::open_with(path, &open_options(options)).map_err(failure)?;
        Ok(Self { inner: Some(inner) })
    }

    /// Open an existing `.rr` file without a writer claim. Reads see the file
    /// as it was when opened.
    #[napi(factory)]
    pub fn open_read_only(path: String) -> Result<Self> {
        let inner = rustyred::Database::open_read_only(path).map_err(failure)?;
        Ok(Self { inner: Some(inner) })
    }

    /// An empty graph with no file and no durability.
    #[napi(factory)]
    pub fn in_memory() -> Self {
        Self {
            inner: Some(rustyred::Database::in_memory()),
        }
    }

    /// Release the file and its writer claim. Later calls throw `database_closed`.
    #[napi]
    pub fn close(&mut self) {
        self.inner = None;
    }

    #[napi(getter)]
    pub fn path(&self) -> Result<Option<String>> {
        Ok(self.handle()?.path().map(|path| path.display().to_string()))
    }

    #[napi(getter)]
    pub fn writable(&self) -> Result<bool> {
        Ok(self.handle()?.is_writable())
    }

    #[napi]
    pub fn inspect(&self) -> Result<Inspection> {
        let inspection = self.handle()?.inspect().map_err(failure)?;
        Ok(Inspection {
            path: inspection.path,
            database_id: inspection.database_id,
            writable: inspection.writable,
            version: inspection.version as i64,
            nodes: inspection.nodes as i64,
            edges: inspection.edges as i64,
        })
    }

    /// Check the recovered graph without writing.
    #[napi(
        ts_return_type = "{ ok: boolean; stats: Record<string, unknown>; problems: Array<unknown> }"
    )]
    pub fn verify(&self) -> Result<Value> {
        let report = self.handle()?.verify().map_err(failure)?;
        serde_json::to_value(report).map_err(|error| {
            napi::Error::new("verify_report_unserializable".into(), error.to_string())
        })
    }

    #[napi]
    pub fn mutate(&mut self, batch: Batch) -> Result<Commit> {
        let mutations = batch
            .delete_edges
            .into_iter()
            .flatten()
            .map(GraphMutation::EdgeDelete)
            .chain(
                batch
                    .delete_nodes
                    .into_iter()
                    .flatten()
                    .map(GraphMutation::NodeDelete),
            )
            .chain(batch.upsert_nodes.into_iter().flatten().map(|node| {
                GraphMutation::NodeUpsert(NodeRecord::new(node.id, node.labels, node.properties))
            }))
            .chain(batch.upsert_edges.into_iter().flatten().map(|edge| {
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
            .mutate(GraphMutationBatch::new(mutations))
            .map_err(failure)?;
        Ok(Commit {
            txn_id: transaction.txn_id as i64,
            graph_version: transaction.graph_version as i64,
        })
    }

    #[napi]
    pub fn get_node(&self, id: String) -> Result<Option<Node>> {
        Ok(self.live_node(&id)?.map(Node::from))
    }

    #[napi]
    pub fn get_edge(&self, id: String) -> Result<Option<Edge>> {
        let edge = self.handle()?.engine().get_edge(&id).map_err(storage)?;
        Ok(edge.filter(|edge| !edge.tombstone).map(Edge::from))
    }

    #[napi]
    pub fn query_nodes(&self, filter: NodeFilter) -> Result<Vec<Node>> {
        let properties = match filter.properties {
            None => BTreeMap::new(),
            Some(Value::Object(map)) => map.into_iter().collect(),
            Some(_) => {
                return Err(napi::Error::new(
                    "invalid_node_filter".into(),
                    "properties must be an object",
                ))
            }
        };
        let limit = filter.limit.map(|limit| limit as usize);
        let Some(prefix) = filter.prefix else {
            let query = NodeQuery {
                label: filter.label,
                properties,
                limit,
                include_expired: false,
            };
            let nodes = self
                .handle()?
                .engine()
                .query_nodes(query)
                .map_err(storage)?;
            return Ok(nodes.into_iter().map(Node::from).collect());
        };
        let engine = self.handle()?.engine();
        let ids = match &filter.label {
            Some(label) => engine.node_ids_for_label_and_string_property_prefix(
                label,
                &prefix.property,
                &prefix.value,
            ),
            None => engine.node_ids_for_string_property_prefix(&prefix.property, &prefix.value),
        }
        .map_err(storage)?;
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
        Ok(nodes)
    }

    #[napi]
    pub fn neighbors(&self, filter: NeighborFilter) -> Result<Vec<Neighbor>> {
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
        for hit in engine.neighbors(query).map_err(storage)? {
            let Some(edge) = engine.get_edge(&hit.edge_id).map_err(storage)? else {
                continue;
            };
            if edge.tombstone {
                continue;
            }
            neighbors.push(Neighbor {
                edge_id: hit.edge_id,
                node_id: hit.node_id,
                edge_type: hit.edge_type,
                properties: edge.properties,
            });
        }
        Ok(neighbors)
    }

    /// Run one read-only Cypher query with bound parameters. `rowCap`
    /// defaults to 1000; a result cut there is `truncated`.
    #[napi]
    pub fn query(
        &self,
        cypher: String,
        params: Option<HashMap<String, Value>>,
        row_cap: Option<u32>,
    ) -> Result<QueryResult> {
        let params = params.unwrap_or_default().into_iter().collect();
        let result = self
            .handle()?
            .query_with_limit(&cypher, &params, row_cap.unwrap_or(1_000) as usize)
            .map_err(failure)?;
        Ok(QueryResult {
            columns: result.columns,
            rows: result.rows,
            truncated: result.truncated,
        })
    }

    /// Keep a full-text index of `fields` (field name to boost) of every node
    /// labelled `label`. The index is saved with each commit and read back on
    /// open. Designating the same fields again changes nothing.
    #[napi]
    pub fn designate_fulltext(
        &mut self,
        label: String,
        fields: HashMap<String, f64>,
    ) -> Result<()> {
        let mut fields = fields
            .iter()
            .map(|(name, boost)| (name.as_str(), *boost as f32))
            .collect::<Vec<_>>();
        fields.sort_by(|a, b| a.0.cmp(b.0));
        self.handle_mut()?
            .designate_fulltext(&label, &fields)
            .map_err(failure)
    }

    /// Rank `label` nodes by BM25 over the designated fields. Every word must
    /// match, the last also as the start of a longer word. `within` keeps only
    /// the given node ids.
    #[napi]
    pub fn fulltext_search(
        &self,
        label: String,
        query: String,
        limit: u32,
        within: Option<Vec<String>>,
    ) -> Result<Vec<FullTextHit>> {
        let hits = self
            .handle()?
            .fulltext_search(&label, &query, limit as usize, within.as_deref())
            .map_err(failure)?;
        Ok(hits
            .into_iter()
            .map(|hit| FullTextHit {
                id: hit.id,
                score: f64::from(hit.score),
                snippet: hit.snippet,
            })
            .collect())
    }

    /// Rewrite the file without the chunks no committed root reaches.
    #[napi]
    pub fn compact(&mut self) -> Result<CompactionReport> {
        let report = self.handle_mut()?.compact().map_err(failure)?;
        Ok(CompactionReport {
            chunks_before: report.chunks_before as i64,
            chunks_after: report.chunks_after as i64,
            chunks_removed: report.chunks_removed as i64,
            bytes_before: report.bytes_before as i64,
            bytes_after: report.bytes_after as i64,
            bytes_removed: report.bytes_removed as i64,
        })
    }
}
