use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::access_method::{
    AccessMethodCache, AccessMethodRegistry, AccessMethodRow, Predicate, ScalarValue,
};
use crate::graph::capability::GraphRead;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::record::NodeRecord;

const MEMORY_DOCUMENT_LABEL: &str = "MemoryDocument";
const MEMORY_DOCUMENT_RELATION: &str = "memorydocument";
const TENANT_COLUMN: &str = "tenant_slug";
const STATUS_COLUMN: &str = "status";
const UPDATED_AT_COLUMN: &str = "updated_at";
const TENANT_UPDATED_AT_COLUMN: &str = "tenant_updated_at";
const TENANT_STATUS_UPDATED_AT_COLUMN: &str = "tenant_status_updated_at";
const MAX_VERSION_RETRIES: usize = 3;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct MemoryDocumentQuery {
    pub tenant_slug: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub since: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub before: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
    #[serde(default)]
    pub include_deleted: bool,
}

impl MemoryDocumentQuery {
    pub fn tenant(tenant_slug: impl Into<String>) -> Self {
        Self {
            tenant_slug: tenant_slug.into(),
            ..Self::default()
        }
    }

    pub fn with_status(mut self, status: impl Into<String>) -> Self {
        let status = status.into();
        if !status.trim().is_empty() {
            self.status = Some(status);
        }
        self
    }

    pub fn with_since(mut self, since: impl Into<String>) -> Self {
        let since = since.into();
        if !since.trim().is_empty() {
            self.since = Some(since);
        }
        self
    }

    pub fn with_before(mut self, before: impl Into<String>) -> Self {
        let before = before.into();
        if !before.trim().is_empty() {
            self.before = Some(before);
        }
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        if limit > 0 {
            self.limit = Some(limit);
        }
        self
    }

    fn normalized_tenant_slug(&self) -> String {
        self.tenant_slug.trim().to_string()
    }

    fn normalized_status(&self) -> Option<String> {
        self.status
            .as_deref()
            .map(str::trim)
            .filter(|status| !status.is_empty())
            .map(str::to_string)
    }

    fn since(&self) -> &str {
        self.since.as_deref().map(str::trim).unwrap_or_default()
    }

    fn before(&self) -> &str {
        self.before.as_deref().map(str::trim).unwrap_or_default()
    }

    fn bounded_limit(&self) -> usize {
        self.limit.filter(|limit| *limit > 0).unwrap_or(100)
    }
}

/// Application-level graph lens for ordered memory-document reads.
///
/// The graph remains the only value authority. The registered ordered access
/// method retains lexicographic `updated_at` keys and node ids, while every
/// returned record is resolved from `GraphRead` after the index scan.
pub struct MemoryDocumentLens<'g, G: GraphRead> {
    graph: &'g G,
    access_methods: Arc<AccessMethodRegistry>,
}

impl<'g, G: GraphRead> MemoryDocumentLens<'g, G> {
    pub fn bind(graph: &'g G) -> GraphStoreResult<Self> {
        Self::bind_with_registry(graph, AccessMethodCache::ordered().registry())
    }

    pub fn bind_with_registry(
        graph: &'g G,
        access_methods: Arc<AccessMethodRegistry>,
    ) -> GraphStoreResult<Self> {
        ensure_index_current(graph, &access_methods)?;
        Ok(Self {
            graph,
            access_methods,
        })
    }

    pub fn query(&self, query: MemoryDocumentQuery) -> GraphStoreResult<Vec<NodeRecord>> {
        let tenant_slug = query.normalized_tenant_slug();
        if tenant_slug.is_empty() {
            return Ok(Vec::new());
        }
        let since = query.since();
        let before = query.before();
        if !since.is_empty() && !before.is_empty() && since >= before {
            return Ok(Vec::new());
        }
        let requested_status = query.normalized_status();
        for _ in 0..MAX_VERSION_RETRIES {
            let graph_version = ensure_index_current(self.graph, &self.access_methods)?;
            let (column, prefix) = if let Some(status) = requested_status.as_deref() {
                (
                    TENANT_STATUS_UPDATED_AT_COLUMN,
                    composite_prefix(&[tenant_slug.as_str(), status]),
                )
            } else {
                (
                    TENANT_UPDATED_AT_COLUMN,
                    composite_prefix(&[tenant_slug.as_str()]),
                )
            };
            let ordered = self
                .access_methods
                .methods()
                .find(|method| method.name() == "ordered")
                .ok_or_else(|| {
                    GraphStoreError::new(
                        "memory_document_ordered_index_missing",
                        "registered memory-document ordered access method is missing",
                    )
                })?;
            let ordered_ids = ordered
                .scan(
                    MEMORY_DOCUMENT_RELATION,
                    &Predicate::Prefix {
                        column: column.to_string(),
                        prefix,
                    },
                )?
                .rows;
            let mut nodes = Vec::with_capacity(query.bounded_limit().min(ordered_ids.len()));
            for node_id in ordered_ids.into_iter().rev() {
                let Some(node) = self.graph.get_node(&node_id).cloned() else {
                    continue;
                };
                if normalized_string_property(&node.properties, TENANT_COLUMN).as_deref()
                    != Some(tenant_slug.as_str())
                {
                    continue;
                }
                let status = normalized_status(&node.properties);
                if requested_status
                    .as_deref()
                    .is_some_and(|requested| requested != status)
                    || (!query.include_deleted && status == "deleted")
                {
                    continue;
                }
                let updated_at = normalized_string_property(&node.properties, UPDATED_AT_COLUMN)
                    .unwrap_or_default();
                if (!since.is_empty() && updated_at.as_str() < since)
                    || (!before.is_empty() && updated_at.as_str() >= before)
                {
                    continue;
                }
                nodes.push(node);
                if nodes.len() >= query.bounded_limit() {
                    break;
                }
            }
            if self.graph.graph_version() == graph_version
                && self.access_methods.is_current(graph_version)
            {
                return Ok(nodes);
            }
            self.access_methods.invalidate();
        }
        Err(GraphStoreError::new(
            "memory_document_graph_changed",
            "graph version changed repeatedly during the memory-document query",
        ))
    }
}

pub fn query_memory_documents<G: GraphRead>(
    graph: &G,
    query: MemoryDocumentQuery,
) -> GraphStoreResult<Vec<NodeRecord>> {
    MemoryDocumentLens::bind(graph)?.query(query)
}

fn normalized_string_property(properties: &Value, key: &str) -> Option<String> {
    properties
        .get(key)?
        .as_str()
        .map(str::trim)
        .map(str::to_string)
}

fn normalized_status(properties: &Value) -> String {
    normalized_string_property(properties, STATUS_COLUMN)
        .filter(|status| !status.is_empty())
        .unwrap_or_else(|| "active".to_string())
}

fn ensure_index_current<G: GraphRead>(
    graph: &G,
    access_methods: &AccessMethodRegistry,
) -> GraphStoreResult<u64> {
    for _ in 0..MAX_VERSION_RETRIES {
        let graph_version = graph.graph_version();
        if access_methods.is_current(graph_version) {
            return Ok(graph_version);
        }
        let rows = graph
            .node_ids_for_label(MEMORY_DOCUMENT_LABEL)
            .into_iter()
            .filter_map(|row_id| memory_document_access_row(graph, row_id))
            .collect::<Vec<_>>();
        if graph.graph_version() != graph_version {
            continue;
        }
        access_methods.rebuild_from_graph_rows(&rows, graph_version)?;
        if graph.graph_version() == graph_version {
            return Ok(graph_version);
        }
        access_methods.invalidate();
    }
    Err(GraphStoreError::new(
        "memory_document_graph_changed",
        "graph version changed repeatedly while rebuilding the memory-document index",
    ))
}

fn memory_document_access_row<G: GraphRead>(graph: &G, row_id: String) -> Option<AccessMethodRow> {
    let node = graph.get_node(&row_id)?;
    let tenant = normalized_string_property(&node.properties, TENANT_COLUMN)?;
    if tenant.is_empty() {
        return None;
    }
    let status = normalized_status(&node.properties);
    let updated_at =
        normalized_string_property(&node.properties, UPDATED_AT_COLUMN).unwrap_or_default();
    Some(AccessMethodRow {
        relation: MEMORY_DOCUMENT_RELATION.to_string(),
        row_id,
        values: BTreeMap::from([
            (
                TENANT_UPDATED_AT_COLUMN.to_string(),
                ScalarValue::String(format!(
                    "{}{}",
                    composite_prefix(&[tenant.as_str()]),
                    updated_at
                )),
            ),
            (
                TENANT_STATUS_UPDATED_AT_COLUMN.to_string(),
                ScalarValue::String(format!(
                    "{}{}",
                    composite_prefix(&[tenant.as_str(), status.as_str()]),
                    updated_at
                )),
            ),
        ]),
    })
}

fn composite_prefix(parts: &[&str]) -> String {
    let mut prefix = String::new();
    for part in parts {
        prefix.push_str(&part.len().to_string());
        prefix.push(':');
        prefix.push_str(part);
        prefix.push('|');
    }
    prefix
}
