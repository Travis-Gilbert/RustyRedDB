use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, DerefMut};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::access_method::{
    AccessMethodRegistry, AccessMethodRow, ColumnId, RankingRegistry, RelationId, RowChange,
    RowChangeKind, RowId, ScalarValue,
};
use crate::cold_fragments::{is_epistemic_column, HotColumnGroup};
use crate::graph::capability::{GraphRead, GraphWrite};
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph::in_memory::InMemoryGraphStore;
use crate::graph::record::{EdgeRecord, NodeRecord};
pub use crate::resolved_row::ResolvedRow;

const RELATIONAL_ROW_ID_PROPERTY: &str = "_rustyred_relational_row_id";
const RELATIONAL_RELATION_PROPERTY: &str = "_rustyred_relational_relation";
const RELATIONAL_NODE_PREFIX: &str = "_rustyred_relational_node:";
const LABEL_RELATION_NAMESPACE: &str = "label";
const EDGE_TYPE_RELATION_NAMESPACE: &str = "edge";
const NATIVE_CATALOG_NODE_PREFIX: &str = "_rustyred_native_catalog:";
const NATIVE_TENANT_LABEL: &str = "_rustyred_native_tenant";
const NATIVE_PROJECT_LABEL: &str = "_rustyred_native_project";
const NATIVE_BILLING_LABEL: &str = "_rustyred_native_billing_account";
const NATIVE_AUTH_PRINCIPAL_LABEL: &str = "_rustyred_native_auth_principal";

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ColumnSchema {
    pub name: ColumnId,
    pub nullable: bool,
    pub indexed: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RelationSchema {
    pub relation: RelationId,
    pub columns: Vec<ColumnSchema>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum RelationSource {
    AllNodes,
    AllEdges,
    Label(String),
    EdgeType(String),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RelationDefinition {
    pub relation: RelationId,
    pub source: RelationSource,
    pub columns: Vec<ColumnSchema>,
}

/// A zero-row view over one relation definition and a live graph handle.
pub struct ResolvedRelation<'g, G: GraphRead> {
    pub schema: RelationSchema,
    source: &'g RelationSource,
    graph: &'g G,
}

impl<G: GraphRead> ResolvedRelation<'_, G> {
    pub fn get(&self, id: &str) -> Option<ResolvedRow> {
        match self.source {
            RelationSource::AllNodes => self
                .graph
                .get_node(id)
                .map(|node| resolved_node(&self.schema.relation, node)),
            RelationSource::AllEdges => self
                .graph
                .get_edge(id)
                .map(|edge| resolved_edge(&self.schema.relation, edge)),
            RelationSource::Label(label) => {
                let storage_id =
                    relational_node_storage_id(self.graph, &self.schema.relation, label, id)?;
                self.graph
                    .get_node(&storage_id)
                    .map(|node| resolved_node(&self.schema.relation, node))
            }
            RelationSource::EdgeType(edge_type) => self.graph.get_edge(id).and_then(|edge| {
                edge.edge_type
                    .eq_ignore_ascii_case(edge_type)
                    .then(|| resolved_edge(&self.schema.relation, edge))
            }),
        }
    }

    pub fn rows(&self) -> impl Iterator<Item = ResolvedRow> + '_ {
        self.row_ids()
            .into_iter()
            .filter_map(move |id| self.get(&id))
    }

    pub fn row_ids(&self) -> Vec<RowId> {
        let mut row_ids = match self.source {
            RelationSource::AllNodes => self
                .graph
                .live_node_records()
                .into_iter()
                .map(|node| node.id)
                .collect(),
            RelationSource::AllEdges => self
                .graph
                .live_edge_records()
                .into_iter()
                .map(|edge| edge.id)
                .collect(),
            RelationSource::Label(label) => self
                .graph
                .node_ids_for_label_folded(label)
                .into_iter()
                .filter(|id| self.graph.get_node(id).is_some())
                .collect(),
            RelationSource::EdgeType(edge_type) => self.graph.edge_ids_for_type_folded(edge_type),
        };
        row_ids.sort();
        row_ids
    }
}

/// A relational address space over the graph. The generic handle may be an
/// owned graph (for embedded/test use) or a borrowed `&G` (for live callers).
/// No resolved rows or property values are reachable from this type.
pub struct RelationalLens<G: GraphRead = InMemoryGraphStore> {
    definitions: BTreeMap<RelationId, RelationDefinition>,
    access_methods: AccessMethodRegistry,
    ranking_methods: RankingRegistry,
    graph: G,
}

/// Scoped direct access to a relational lens's graph. Graph-derived relation
/// definitions and access methods are refreshed when the mutable borrow ends.
pub struct RelationalGraphMut<'a, G: GraphRead> {
    graph: &'a mut G,
    definitions: &'a mut BTreeMap<RelationId, RelationDefinition>,
    access_methods: &'a AccessMethodRegistry,
}

impl<G: GraphRead> Deref for RelationalGraphMut<'_, G> {
    type Target = G;

    fn deref(&self) -> &Self::Target {
        self.graph
    }
}

impl<G: GraphRead> DerefMut for RelationalGraphMut<'_, G> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.graph
    }
}

impl<G: GraphRead> Drop for RelationalGraphMut<'_, G> {
    fn drop(&mut self) {
        refresh_definitions_after_graph_mutation(self.graph, self.definitions);
        self.access_methods.invalidate();
    }
}

impl Default for RelationalLens<InMemoryGraphStore> {
    fn default() -> Self {
        Self::new()
    }
}

impl RelationalLens<InMemoryGraphStore> {
    pub fn new() -> Self {
        Self::from_graph_unindexed(
            InMemoryGraphStore::new(),
            AccessMethodRegistry::with_native_defaults(),
        )
    }

    pub fn with_access_methods(access_methods: AccessMethodRegistry) -> Self {
        Self::from_graph_unindexed(InMemoryGraphStore::new(), access_methods)
    }
}

impl<'g, G: GraphRead> RelationalLens<&'g G> {
    pub fn bind(graph: &'g G) -> GraphStoreResult<Self> {
        Self::from_graph(graph)
    }
}

impl<G: GraphRead> RelationalLens<G> {
    pub fn from_graph(graph: G) -> GraphStoreResult<Self> {
        let lens = Self::from_graph_unindexed(graph, AccessMethodRegistry::with_native_defaults());
        lens.rebuild_access_methods()?;
        Ok(lens)
    }

    pub fn from_graph_with_access_methods(
        graph: G,
        access_methods: AccessMethodRegistry,
    ) -> GraphStoreResult<Self> {
        let lens = Self::from_graph_unindexed(graph, access_methods);
        lens.rebuild_access_methods()?;
        Ok(lens)
    }

    fn from_graph_unindexed(graph: G, access_methods: AccessMethodRegistry) -> Self {
        Self {
            definitions: discover_definitions(&graph),
            access_methods,
            ranking_methods: RankingRegistry::with_native_defaults(),
            graph,
        }
    }

    pub fn graph(&self) -> &G {
        &self.graph
    }

    pub fn access_methods(&self) -> &AccessMethodRegistry {
        &self.access_methods
    }

    pub fn ranking_methods(&self) -> &RankingRegistry {
        &self.ranking_methods
    }

    pub fn definitions(&self) -> impl Iterator<Item = &RelationDefinition> {
        self.definitions.values()
    }

    pub fn relation(&self, relation: &str) -> Option<ResolvedRelation<'_, G>> {
        let relation = normalize_relation_id(relation);
        self.definitions
            .get(&relation)
            .map(|definition| ResolvedRelation {
                schema: RelationSchema {
                    relation: definition.relation.clone(),
                    columns: definition.columns.clone(),
                },
                source: &definition.source,
                graph: &self.graph,
            })
    }

    pub fn relations(&self) -> impl Iterator<Item = ResolvedRelation<'_, G>> {
        self.definitions
            .values()
            .map(|definition| ResolvedRelation {
                schema: RelationSchema {
                    relation: definition.relation.clone(),
                    columns: definition.columns.clone(),
                },
                source: &definition.source,
                graph: &self.graph,
            })
    }

    pub fn hot_column_group(&self, relation: &str) -> Option<HotColumnGroup> {
        let relation = self.relation(relation)?;
        let rows = relation.rows().collect::<Vec<_>>();
        let has_epistemic_columns = rows
            .iter()
            .any(|row| row.values.keys().any(|column| is_epistemic_column(column)));
        if !has_epistemic_columns {
            return None;
        }
        let group = HotColumnGroup::epistemic(relation.schema.relation, &rows);
        (!group.is_empty()).then_some(group)
    }

    fn rebuild_access_methods(&self) -> GraphStoreResult<()> {
        let mut rows = Vec::new();
        for relation in self.relations() {
            rows.extend(relation.rows().map(|row| AccessMethodRow {
                relation: row.relation,
                row_id: row.id,
                values: row.values,
            }));
        }
        self.access_methods
            .rebuild_from_graph_rows(&rows, self.graph.graph_version())
    }

    pub fn ensure_access_methods_current(&self) -> GraphStoreResult<()> {
        let graph_version = self.graph.graph_version();
        if !self.access_methods.is_current(graph_version) {
            self.rebuild_access_methods()?;
        }
        Ok(())
    }
}

impl<G: GraphRead + GraphWrite> RelationalLens<G> {
    pub fn graph_mut(&mut self) -> RelationalGraphMut<'_, G> {
        RelationalGraphMut {
            graph: &mut self.graph,
            definitions: &mut self.definitions,
            access_methods: &self.access_methods,
        }
    }

    pub fn create_relation(&mut self, schema: RelationSchema) {
        let relation = normalize_relation_id(&schema.relation);
        self.definitions
            .entry(relation.clone())
            .or_insert(RelationDefinition {
                relation: relation.clone(),
                source: source_for_relation(&relation),
                columns: schema.columns,
            });
    }

    pub fn upsert_row(&mut self, mut row: ResolvedRow) -> GraphStoreResult<()> {
        validate_resolved_row(&row)?;
        row.relation = normalize_relation_id(&row.relation);
        let relation_id = row.relation.clone();
        let existing = self
            .relation(&relation_id)
            .and_then(|relation| relation.get(&row.id));
        let existing_storage_id = existing.as_ref().map(|existing| existing.id.clone());
        if let Some(existing) = &existing {
            self.access_methods.on_write(&RowChange {
                relation: relation_id.clone(),
                row_id: existing.id.clone(),
                values: existing.values.clone(),
                properties: existing.properties.clone(),
                kind: RowChangeKind::Delete,
            })?;
        }
        self.definitions
            .entry(relation_id.clone())
            .or_insert_with(|| definition_for_row(&row));
        merge_definition_columns(
            self.definitions
                .get_mut(&relation_id)
                .expect("definition inserted above"),
            row.values.keys().cloned(),
        );
        let source = self
            .definitions
            .get(&relation_id)
            .map(|definition| definition.source.clone())
            .expect("definition inserted above");
        let stored_row_id = match &source {
            RelationSource::AllEdges | RelationSource::EdgeType(_) => {
                let edge = row_as_edge(&self.graph, &source, &row)?;
                let stored_row_id = edge.id.clone();
                self.graph.upsert_edge(edge)?;
                stored_row_id
            }
            RelationSource::AllNodes | RelationSource::Label(_) => {
                let node = row_as_node(&self.graph, &source, &row, existing_storage_id.as_deref());
                let stored_row_id = node.id.clone();
                self.graph.upsert_node(node)?;
                stored_row_id
            }
        };
        self.access_methods.on_write(&RowChange {
            relation: relation_id.clone(),
            row_id: stored_row_id,
            values: row.values.clone(),
            properties: row.properties.clone(),
            kind: RowChangeKind::Upsert,
        })?;
        self.access_methods.mark_current(self.graph.graph_version());
        Ok(())
    }

    pub fn delete_row(&mut self, relation: &str, row_id: &str) -> GraphStoreResult<bool> {
        let relation = normalize_relation_id(relation);
        let Some(existing) = self
            .relation(&relation)
            .and_then(|relation| relation.get(row_id))
        else {
            return Ok(false);
        };
        self.access_methods.on_write(&RowChange {
            relation: relation.clone(),
            row_id: existing.id.clone(),
            values: existing.values,
            properties: existing.properties,
            kind: RowChangeKind::Delete,
        })?;
        let source = self
            .definitions
            .get(&relation)
            .map(|definition| definition.source.clone())
            .expect("resolved relation has a definition");
        match source {
            RelationSource::AllNodes => {
                if let Some(node) = self.graph.get_node_record(row_id) {
                    tombstone_node_and_incident_edges(&mut self.graph, node)?;
                }
            }
            RelationSource::Label(label) => {
                if let Some(storage_id) =
                    relational_node_storage_id(&self.graph, &relation, &label, row_id)
                {
                    let Some(mut node) = self.graph.get_node_record(&storage_id) else {
                        return Ok(false);
                    };
                    node.labels
                        .retain(|candidate| !candidate.eq_ignore_ascii_case(&label));
                    if node.labels.is_empty() {
                        tombstone_node_and_incident_edges(&mut self.graph, node)?;
                    } else {
                        self.graph.upsert_node(node)?;
                    }
                }
            }
            RelationSource::AllEdges | RelationSource::EdgeType(_) => {
                if let Some(mut edge) = self.graph.get_edge_record(row_id) {
                    edge.tombstone = true;
                    self.graph.upsert_edge(edge)?;
                }
            }
        }
        self.access_methods.invalidate();
        Ok(true)
    }

    pub fn drop_relation(&mut self, relation: &str) -> GraphStoreResult<bool> {
        let relation = normalize_relation_id(relation);
        let Some(source) = self
            .definitions
            .get(&relation)
            .map(|definition| definition.source.clone())
        else {
            return Ok(false);
        };
        if matches!(source, RelationSource::AllNodes) {
            return Err(GraphStoreError::new(
                "relational_drop_unsupported_source",
                "graph-backed relation nodes cannot be dropped",
            ));
        }
        let row_ids = self
            .relation(&relation)
            .expect("definition resolved above")
            .row_ids();
        for row_id in row_ids {
            self.delete_row(&relation, &row_id)?;
        }
        self.definitions.remove(&relation);
        Ok(true)
    }

    pub fn replace_relation(
        &mut self,
        schema: RelationSchema,
        rows: Vec<ResolvedRow>,
    ) -> GraphStoreResult<()> {
        self.drop_relation(&schema.relation)?;
        self.create_relation(schema);
        for row in rows {
            self.upsert_row(row)?;
        }
        Ok(())
    }

    pub fn rename_relation(&mut self, from: &str, to: &str) -> GraphStoreResult<bool> {
        let from = normalize_relation_id(from);
        let to = normalize_relation_id(to);
        let Some(existing) = self.relation(&from) else {
            return Ok(false);
        };
        let source = self
            .definitions
            .get(&from)
            .map(|definition| definition.source.clone())
            .expect("resolved relation has a definition");
        if let RelationSource::Label(label) = source {
            let mut schema = existing.schema.clone();
            let rows = existing.rows().collect::<Vec<_>>();
            let destination = to.clone();
            if self
                .definitions
                .get(&destination)
                .is_some_and(|definition| !matches!(definition.source, RelationSource::Label(_)))
            {
                return Err(GraphStoreError::new(
                    "relational_rename_source_conflict",
                    format!("relation {to} is not a label relation"),
                ));
            }
            let mut renamed_nodes = Vec::with_capacity(rows.len());
            for row in &rows {
                let Some(mut node) = self.graph.get_node_record(&row.id) else {
                    return Err(GraphStoreError::new(
                        "relational_rename_missing_row",
                        format!("row {} disappeared while renaming {from} to {to}", row.id),
                    ));
                };
                let mut renamed = false;
                for candidate in &mut node.labels {
                    if candidate.eq_ignore_ascii_case(&label) {
                        *candidate = destination.clone();
                        renamed = true;
                    }
                }
                if !renamed {
                    return Err(GraphStoreError::new(
                        "relational_rename_missing_label",
                        format!("row {} no longer carries relation label {label}", row.id),
                    ));
                }
                node.labels.sort();
                node.labels.dedup();
                if let Some(properties) = node.properties.as_object_mut() {
                    if properties.contains_key(RELATIONAL_RELATION_PROPERTY) {
                        properties.insert(
                            RELATIONAL_RELATION_PROPERTY.to_string(),
                            Value::String(destination.clone()),
                        );
                    }
                }
                renamed_nodes.push(node);
            }
            for node in renamed_nodes {
                self.graph.upsert_node(node)?;
            }
            for row in &rows {
                self.access_methods.on_write(&RowChange {
                    relation: from.clone(),
                    row_id: row.id.clone(),
                    values: row.values.clone(),
                    properties: row.properties.clone(),
                    kind: RowChangeKind::Delete,
                })?;
            }
            self.definitions.remove(&from);
            schema.relation = destination.clone();
            if let Some(definition) = self.definitions.get_mut(&destination) {
                merge_definition_columns(
                    definition,
                    schema.columns.iter().map(|column| column.name.clone()),
                );
            } else {
                self.definitions.insert(
                    destination.clone(),
                    RelationDefinition {
                        relation: destination.clone(),
                        source: RelationSource::Label(destination.clone()),
                        columns: schema.columns,
                    },
                );
            }
            for row in rows {
                let Some(renamed) = self
                    .relation(&destination)
                    .and_then(|relation| relation.get(&row.id))
                else {
                    return Err(GraphStoreError::new(
                        "relational_rename_missing_row",
                        format!(
                            "row {} did not resolve after renaming {from} to {to}",
                            row.id
                        ),
                    ));
                };
                self.access_methods.on_write(&RowChange {
                    relation: destination.clone(),
                    row_id: renamed.id,
                    values: renamed.values,
                    properties: renamed.properties,
                    kind: RowChangeKind::Upsert,
                })?;
            }
            self.access_methods.mark_current(self.graph.graph_version());
            return Ok(true);
        }
        Err(GraphStoreError::new(
            "relational_rename_unsupported_source",
            format!("graph-backed relation {from} cannot be renamed"),
        ))
    }
}

fn discover_definitions(graph: &impl GraphRead) -> BTreeMap<RelationId, RelationDefinition> {
    let nodes = graph.live_node_records();
    let edges = graph.live_edge_records();
    let label_relations = nodes
        .iter()
        .flat_map(|node| node.labels.iter())
        .filter(|label| !is_reserved_relation_id(label))
        .map(|label| normalize_relation_id(label))
        .collect::<BTreeSet<_>>();
    let edge_type_relations = edges
        .iter()
        .map(|edge| &edge.edge_type)
        .filter(|edge_type| !is_reserved_relation_id(edge_type))
        .map(|edge_type| normalize_relation_id(edge_type))
        .collect::<BTreeSet<_>>();
    let colliding_relations = label_relations
        .intersection(&edge_type_relations)
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut occupied_relations = label_relations
        .union(&edge_type_relations)
        .cloned()
        .chain(["nodes".to_string(), "edges".to_string()])
        .collect::<BTreeSet<_>>();
    let mut label_aliases = BTreeMap::new();
    let mut edge_type_aliases = BTreeMap::new();
    for relation in &colliding_relations {
        label_aliases.insert(
            relation.clone(),
            unique_namespaced_relation_id(
                LABEL_RELATION_NAMESPACE,
                relation,
                &mut occupied_relations,
            ),
        );
        edge_type_aliases.insert(
            relation.clone(),
            unique_namespaced_relation_id(
                EDGE_TYPE_RELATION_NAMESPACE,
                relation,
                &mut occupied_relations,
            ),
        );
    }
    let mut definitions = BTreeMap::from([
        (
            "nodes".to_string(),
            RelationDefinition {
                relation: "nodes".to_string(),
                source: RelationSource::AllNodes,
                columns: vec![
                    column("id", true),
                    column("label", true),
                    column("content_hash", false),
                ],
            },
        ),
        (
            "edges".to_string(),
            RelationDefinition {
                relation: "edges".to_string(),
                source: RelationSource::AllEdges,
                columns: vec![
                    column("id", true),
                    column("from_id", true),
                    column("to_id", true),
                    column("type", true),
                ],
            },
        ),
    ]);
    for node in &nodes {
        let property_columns = scalar_property_columns(&node.properties);
        if let Some(definition) = definitions.get_mut("nodes") {
            merge_definition_columns(definition, property_columns.clone());
        }
        for label in &node.labels {
            if is_reserved_relation_id(label) {
                continue;
            }
            let source_relation = normalize_relation_id(label);
            let relation = label_aliases
                .get(&source_relation)
                .cloned()
                .unwrap_or_else(|| source_relation.clone());
            let definition =
                definitions
                    .entry(relation.clone())
                    .or_insert_with(|| RelationDefinition {
                        relation: relation.clone(),
                        source: RelationSource::Label(source_relation.clone()),
                        columns: vec![
                            column("id", true),
                            column("label", true),
                            column("content_hash", false),
                        ],
                    });
            merge_definition_columns(definition, property_columns.clone());
        }
    }
    // CamelCase graph labels also expose snake_case SQL names
    // (`DatasetVersion` → `dataset_version`) without changing the label set.
    let snake_aliases = {
        let mut aliases = Vec::new();
        for node in &nodes {
            for label in &node.labels {
                if is_reserved_relation_id(label) {
                    continue;
                }
                let source_relation = normalize_relation_id(label);
                let primary = label_aliases
                    .get(&source_relation)
                    .cloned()
                    .unwrap_or_else(|| source_relation.clone());
                let snake = camel_label_to_snake(label);
                if snake == primary
                    || occupied_relations.contains(&snake)
                    || definitions.contains_key(&snake)
                {
                    continue;
                }
                if let Some(definition) = definitions.get(&primary) {
                    let mut alias = definition.clone();
                    alias.relation = snake.clone();
                    occupied_relations.insert(snake.clone());
                    aliases.push((snake, alias));
                }
            }
        }
        aliases
    };
    for (snake, alias) in snake_aliases {
        definitions.entry(snake).or_insert(alias);
    }
    for edge in &edges {
        let property_columns = scalar_property_columns(&edge.properties);
        if let Some(definition) = definitions.get_mut("edges") {
            merge_definition_columns(definition, property_columns.clone());
        }
        if is_reserved_relation_id(&edge.edge_type) {
            continue;
        }
        let source_relation = normalize_relation_id(&edge.edge_type);
        let relation = edge_type_aliases
            .get(&source_relation)
            .cloned()
            .unwrap_or_else(|| source_relation.clone());
        let definition =
            definitions
                .entry(relation.clone())
                .or_insert_with(|| RelationDefinition {
                    relation: relation.clone(),
                    source: RelationSource::EdgeType(source_relation.clone()),
                    columns: vec![
                        column("id", true),
                        column("from_id", true),
                        column("to_id", true),
                        column("type", true),
                    ],
                });
        merge_definition_columns(definition, property_columns);
    }
    definitions
}

fn refresh_definitions_after_graph_mutation(
    graph: &impl GraphRead,
    definitions: &mut BTreeMap<RelationId, RelationDefinition>,
) {
    let mut refreshed = discover_definitions(graph);
    for (relation, mut existing) in std::mem::take(definitions) {
        let target_relation = refreshed
            .get(&relation)
            .filter(|discovered| discovered.source == existing.source)
            .map(|_| relation.clone())
            .or_else(|| {
                refreshed
                    .iter()
                    .find(|(_, discovered)| discovered.source == existing.source)
                    .map(|(relation, _)| relation.clone())
            });
        if let Some(target_relation) = target_relation {
            let discovered = refreshed
                .get_mut(&target_relation)
                .expect("target relation was selected from refreshed definitions");
            for column in existing.columns {
                if let Some(current) = discovered
                    .columns
                    .iter_mut()
                    .find(|current| current.name == column.name)
                {
                    *current = column;
                } else {
                    discovered.columns.push(column);
                }
            }
        } else if let RelationSource::Label(label) = &existing.source {
            let relation = if refreshed.contains_key(&relation) {
                let mut occupied = refreshed.keys().cloned().collect::<BTreeSet<_>>();
                unique_namespaced_relation_id(LABEL_RELATION_NAMESPACE, label, &mut occupied)
            } else {
                relation
            };
            existing.relation = relation.clone();
            refreshed.insert(relation, existing);
        }
    }
    *definitions = refreshed;
}

fn unique_namespaced_relation_id(
    namespace: &str,
    relation: &str,
    occupied: &mut BTreeSet<RelationId>,
) -> RelationId {
    let base = format!("{namespace}:{relation}");
    let mut candidate = base.clone();
    let mut suffix = 2usize;
    while occupied.contains(&candidate) {
        candidate = format!("{base}#{suffix}");
        suffix += 1;
    }
    occupied.insert(candidate.clone());
    candidate
}

fn tombstone_node_and_incident_edges(
    graph: &mut (impl GraphRead + GraphWrite),
    mut node: NodeRecord,
) -> GraphStoreResult<()> {
    let incident_edges = graph
        .live_edge_records()
        .into_iter()
        .filter(|edge| edge.from_id == node.id || edge.to_id == node.id)
        .collect::<Vec<_>>();
    for mut edge in incident_edges {
        edge.tombstone = true;
        graph.upsert_edge(edge)?;
    }
    node.tombstone = true;
    graph.upsert_node(node)?;
    Ok(())
}

fn column(name: &str, indexed: bool) -> ColumnSchema {
    ColumnSchema {
        name: name.to_string(),
        nullable: true,
        indexed,
    }
}

fn scalar_property_columns(properties: &Value) -> Vec<ColumnId> {
    properties
        .as_object()
        .into_iter()
        .flat_map(|map| map.iter())
        .filter(|(key, _)| !is_relational_metadata(key))
        .filter_map(|(key, value)| ScalarValue::from_json(value).map(|_| key.clone()))
        .collect()
}

fn merge_definition_columns(
    definition: &mut RelationDefinition,
    columns: impl IntoIterator<Item = ColumnId>,
) {
    let mut known = definition
        .columns
        .iter()
        .map(|column| column.name.clone())
        .collect::<BTreeSet<_>>();
    let mut additions = columns
        .into_iter()
        .filter(|column| known.insert(column.clone()))
        .collect::<Vec<_>>();
    additions.sort();
    definition
        .columns
        .extend(additions.into_iter().map(|name| column(&name, true)));
}

fn resolved_node(relation: &str, node: &NodeRecord) -> ResolvedRow {
    let properties = user_properties(&node.properties);
    let mut values = scalar_values(&properties);
    values.insert("id".to_string(), ScalarValue::String(node.id.clone()));
    values.insert(
        "label".to_string(),
        ScalarValue::String(node.labels.first().cloned().unwrap_or_default()),
    );
    if let Some(content_hash) = &node.content_hash {
        values.insert(
            "content_hash".to_string(),
            ScalarValue::String(content_hash.clone()),
        );
    }
    ResolvedRow {
        id: node.id.clone(),
        relation: relation.to_string(),
        values,
        properties,
    }
}

fn resolved_edge(relation: &str, edge: &EdgeRecord) -> ResolvedRow {
    let mut values = scalar_values(&edge.properties);
    values.extend([
        ("id".to_string(), ScalarValue::String(edge.id.clone())),
        (
            "from_id".to_string(),
            ScalarValue::String(edge.from_id.clone()),
        ),
        ("to_id".to_string(), ScalarValue::String(edge.to_id.clone())),
        (
            "type".to_string(),
            ScalarValue::String(edge.edge_type.clone()),
        ),
    ]);
    ResolvedRow {
        id: edge.id.clone(),
        relation: relation.to_string(),
        values,
        properties: edge.properties.clone(),
    }
}

fn scalar_values(properties: &Value) -> BTreeMap<ColumnId, ScalarValue> {
    properties
        .as_object()
        .into_iter()
        .flat_map(|map| map.iter())
        .filter(|(key, _)| !is_relational_metadata(key))
        .filter_map(|(key, value)| ScalarValue::from_json(value).map(|value| (key.clone(), value)))
        .collect()
}

fn validate_resolved_row(row: &ResolvedRow) -> GraphStoreResult<()> {
    if row.id.trim().is_empty() || row.relation.trim().is_empty() {
        return Err(GraphStoreError::new(
            "invalid_relational_row",
            "relation and row id must be non-empty",
        ));
    }
    Ok(())
}

fn normalize_relation_id(relation: &str) -> RelationId {
    relation.trim().to_ascii_lowercase()
}

/// `DatasetVersion` → `dataset_version`. Already-snake or lowercase labels pass through.
fn camel_label_to_snake(label: &str) -> RelationId {
    let trimmed = label.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.contains('_') || !trimmed.chars().any(|ch| ch.is_ascii_uppercase()) {
        return normalize_relation_id(trimmed);
    }
    let chars = trimmed.chars().collect::<Vec<_>>();
    let mut out = String::with_capacity(trimmed.len() + 4);
    for (index, ch) in chars.iter().copied().enumerate() {
        if ch.is_ascii_uppercase() {
            let previous_is_lower_or_digit = index > 0
                && (chars[index - 1].is_ascii_lowercase() || chars[index - 1].is_ascii_digit());
            let acronym_boundary = index > 0
                && index + 1 < chars.len()
                && chars[index - 1].is_ascii_uppercase()
                && chars[index + 1].is_ascii_lowercase();
            if previous_is_lower_or_digit || acronym_boundary {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

fn is_reserved_relation_id(relation: &str) -> bool {
    relation.eq_ignore_ascii_case("nodes") || relation.eq_ignore_ascii_case("edges")
}

fn source_for_relation(relation: &str) -> RelationSource {
    let relation = normalize_relation_id(relation);
    match relation.as_str() {
        "nodes" => RelationSource::AllNodes,
        "edges" => RelationSource::AllEdges,
        _ => RelationSource::Label(relation),
    }
}

fn definition_for_row(row: &ResolvedRow) -> RelationDefinition {
    let relation = normalize_relation_id(&row.relation);
    RelationDefinition {
        relation: relation.clone(),
        source: source_for_relation(&relation),
        columns: row.values.keys().map(|name| column(name, true)).collect(),
    }
}

fn row_as_node(
    graph: &impl GraphRead,
    source: &RelationSource,
    row: &ResolvedRow,
    existing_storage_id: Option<&str>,
) -> NodeRecord {
    let storage_id = match existing_storage_id {
        Some(storage_id) => storage_id.to_string(),
        None => match source {
            RelationSource::Label(label)
                if graph
                    .get_node(&row.id)
                    .is_some_and(|node| !node_has_label_folded(node, label)) =>
            {
                relational_node_id(&row.relation, &row.id)
            }
            _ => row.id.clone(),
        },
    };
    let mut node = graph.get_node_record(&storage_id).unwrap_or_else(|| {
        NodeRecord::new(&storage_id, Vec::<String>::new(), Value::Object(Map::new()))
    });
    let stored_alias = node
        .properties
        .get(RELATIONAL_ROW_ID_PROPERTY)
        .and_then(Value::as_str)
        .map(str::to_string);
    let stored_relation = node
        .properties
        .get(RELATIONAL_RELATION_PROPERTY)
        .and_then(Value::as_str)
        .map(str::to_string);
    let relationally_encoded = storage_id.starts_with(RELATIONAL_NODE_PREFIX)
        || stored_alias.is_some()
        || storage_id != row.id;
    let relation_metadata = stored_relation
        .filter(|relation| relation.eq_ignore_ascii_case(&row.relation))
        .unwrap_or_else(|| row.relation.clone());
    if let RelationSource::Label(label) = source {
        if !node_has_label_folded(&node, label) {
            node.labels.push(label.clone());
            node.labels.sort();
            node.labels.dedup();
        }
    } else if let Some(label) = row.values.get("label").and_then(ScalarValue::as_str) {
        if !label.is_empty() && !node.labels.iter().any(|candidate| candidate == label) {
            node.labels.push(label.to_string());
            node.labels.sort();
            node.labels.dedup();
        }
    }
    node.properties = merged_properties(
        &row.properties,
        &row.values,
        &["id", "label", "content_hash"],
    );
    if relationally_encoded {
        if let Some(properties) = node.properties.as_object_mut() {
            properties.insert(
                RELATIONAL_ROW_ID_PROPERTY.to_string(),
                Value::String(stored_alias.unwrap_or_else(|| row.id.clone())),
            );
            properties.insert(
                RELATIONAL_RELATION_PROPERTY.to_string(),
                Value::String(relation_metadata),
            );
        }
    }
    node.content_hash = row
        .values
        .get("content_hash")
        .and_then(ScalarValue::as_str)
        .map(str::to_string)
        .or(node.content_hash);
    node.tombstone = false;
    node
}

fn row_as_edge(
    graph: &impl GraphRead,
    source: &RelationSource,
    row: &ResolvedRow,
) -> GraphStoreResult<EdgeRecord> {
    let string_value = |key: &str| {
        row.values
            .get(key)
            .and_then(ScalarValue::as_str)
            .map(str::to_string)
    };
    let existing = graph.get_edge_record(&row.id);
    let from_id = string_value("from_id")
        .or_else(|| existing.as_ref().map(|edge| edge.from_id.clone()))
        .ok_or_else(|| GraphStoreError::new("invalid_relational_edge", "from_id is required"))?;
    let to_id = string_value("to_id")
        .or_else(|| existing.as_ref().map(|edge| edge.to_id.clone()))
        .ok_or_else(|| GraphStoreError::new("invalid_relational_edge", "to_id is required"))?;
    let edge_type = match source {
        RelationSource::EdgeType(edge_type) => existing
            .as_ref()
            .filter(|edge| edge.edge_type.eq_ignore_ascii_case(edge_type))
            .map(|edge| edge.edge_type.clone())
            .unwrap_or_else(|| edge_type.clone()),
        _ => string_value("type")
            .or_else(|| existing.as_ref().map(|edge| edge.edge_type.clone()))
            .ok_or_else(|| GraphStoreError::new("invalid_relational_edge", "type is required"))?,
    };
    let mut edge = existing.unwrap_or_else(|| {
        EdgeRecord::new(
            &row.id,
            &from_id,
            &edge_type,
            &to_id,
            Value::Object(Map::new()),
        )
    });
    edge.from_id = from_id;
    edge.to_id = to_id;
    edge.edge_type = edge_type;
    edge.properties = merged_properties(
        &row.properties,
        &row.values,
        &["id", "from_id", "to_id", "type"],
    );
    edge.tombstone = false;
    Ok(edge)
}

fn merged_properties(
    properties: &Value,
    values: &BTreeMap<ColumnId, ScalarValue>,
    structural: &[&str],
) -> Value {
    let mut map = properties.as_object().cloned().unwrap_or_default();
    for (key, value) in values {
        if structural.contains(&key.as_str()) {
            continue;
        }
        if let Ok(value) = serde_json::to_value(value) {
            map.insert(key.clone(), value);
        }
    }
    Value::Object(map)
}

fn relational_node_id(relation: &str, row_id: &str) -> String {
    format!(
        "{RELATIONAL_NODE_PREFIX}{}:{relation}{row_id}",
        relation.len()
    )
}

fn relational_node_storage_id(
    graph: &impl GraphRead,
    relation: &str,
    label: &str,
    row_id: &str,
) -> Option<String> {
    if graph
        .get_node(row_id)
        .is_some_and(|node| node_has_label_folded(node, label))
    {
        return Some(row_id.to_string());
    }
    let encoded = relational_node_id(relation, row_id);
    if graph
        .get_node(&encoded)
        .is_some_and(|node| node_has_label_folded(node, label))
    {
        return Some(encoded);
    }
    let mut candidates =
        graph.node_ids_for_string_property_exact(RELATIONAL_ROW_ID_PROPERTY, row_id);
    candidates.sort();
    candidates.into_iter().find(|candidate| {
        graph.get_node(candidate).is_some_and(|node| {
            node_has_label_folded(node, label)
                && node
                    .properties
                    .get(RELATIONAL_RELATION_PROPERTY)
                    .and_then(Value::as_str)
                    .is_some_and(|stored_relation| stored_relation.eq_ignore_ascii_case(relation))
        })
    })
}

fn node_has_label_folded(node: &NodeRecord, label: &str) -> bool {
    node.labels
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(label))
}

fn user_properties(properties: &Value) -> Value {
    let Some(properties) = properties.as_object() else {
        return properties.clone();
    };
    let mut properties = properties.clone();
    properties.remove(RELATIONAL_ROW_ID_PROPERTY);
    properties.remove(RELATIONAL_RELATION_PROPERTY);
    Value::Object(properties)
}

fn is_relational_metadata(key: &str) -> bool {
    matches!(
        key,
        RELATIONAL_ROW_ID_PROPERTY | RELATIONAL_RELATION_PROPERTY
    )
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeTenantRecord {
    pub tenant_id: String,
    pub slug: String,
    pub display_name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeProjectRecord {
    pub tenant_id: String,
    pub project_slug: String,
    pub display_name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeBillingAccountRecord {
    pub tenant_id: String,
    pub plan: String,
    pub status: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NativeAuthPrincipalRecord {
    pub principal_id: String,
    pub tenant_id: String,
    pub kind: String,
    pub token_hash: Option<String>,
    pub scopes: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct NativeCatalog<G: GraphRead = InMemoryGraphStore> {
    graph: G,
}

impl Default for NativeCatalog<InMemoryGraphStore> {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeCatalog<InMemoryGraphStore> {
    pub fn new() -> Self {
        Self::from_graph(InMemoryGraphStore::new())
    }
}

impl<G: GraphRead> NativeCatalog<G> {
    pub fn from_graph(graph: G) -> Self {
        Self { graph }
    }

    pub fn graph(&self) -> &G {
        &self.graph
    }

    pub fn into_graph(self) -> G {
        self.graph
    }

    pub fn tenants(&self) -> Vec<NativeTenantRecord> {
        let mut tenants = self.records_with_label(NATIVE_TENANT_LABEL, decode_native_tenant_record);
        tenants.sort_by(|left, right| left.tenant_id.cmp(&right.tenant_id));
        tenants
    }

    pub fn projects_for_tenant(&self, tenant_id: &str) -> Vec<NativeProjectRecord> {
        let mut projects = self.records_for_tenant(
            NATIVE_PROJECT_LABEL,
            tenant_id,
            decode_native_project_record,
        );
        projects.sort_by(|left, right| {
            (&left.tenant_id, &left.project_slug).cmp(&(&right.tenant_id, &right.project_slug))
        });
        projects
    }

    pub fn billing_account_for_tenant(
        &self,
        tenant_id: &str,
    ) -> Option<NativeBillingAccountRecord> {
        let node_id = native_catalog_node_id("billing", &[tenant_id]);
        self.graph
            .get_node(&node_id)
            .and_then(decode_native_billing_record)
    }

    pub fn auth_principals_for_tenant(&self, tenant_id: &str) -> Vec<NativeAuthPrincipalRecord> {
        let mut principals = self.records_for_tenant(
            NATIVE_AUTH_PRINCIPAL_LABEL,
            tenant_id,
            decode_native_auth_principal_record,
        );
        principals.sort_by(|left, right| left.principal_id.cmp(&right.principal_id));
        principals
    }

    fn records_with_label<T>(&self, label: &str, decode: fn(&NodeRecord) -> Option<T>) -> Vec<T> {
        self.graph
            .node_ids_for_label(label)
            .into_iter()
            .filter_map(|node_id| self.graph.get_node(&node_id))
            .filter_map(decode)
            .collect()
    }

    fn records_for_tenant<T>(
        &self,
        label: &str,
        tenant_id: &str,
        decode: fn(&NodeRecord) -> Option<T>,
    ) -> Vec<T> {
        self.graph
            .node_ids_for_string_property_exact("tenant_id", tenant_id)
            .into_iter()
            .filter_map(|node_id| self.graph.get_node(&node_id))
            .filter(|node| node_has_exact_label(node, label))
            .filter_map(decode)
            .collect()
    }

    fn require_tenant(&self, tenant_id: &str) -> GraphStoreResult<()> {
        require_non_empty(tenant_id, "tenant_id")?;
        let node_id = native_catalog_node_id("tenant", &[tenant_id]);
        if !self
            .graph
            .get_node(&node_id)
            .and_then(decode_native_tenant_record)
            .is_some_and(|tenant| tenant.tenant_id == tenant_id)
        {
            return Err(GraphStoreError::new(
                "catalog_foreign_key_violation",
                format!("tenant {tenant_id} does not exist"),
            ));
        }
        Ok(())
    }
}

impl<G: GraphRead + GraphWrite> NativeCatalog<G> {
    pub fn graph_mut(&mut self) -> &mut G {
        &mut self.graph
    }

    pub fn upsert_tenant(&mut self, tenant: NativeTenantRecord) -> GraphStoreResult<()> {
        require_non_empty(&tenant.tenant_id, "tenant_id")?;
        require_non_empty(&tenant.slug, "slug")?;
        let node_id = native_catalog_node_id("tenant", &[&tenant.tenant_id]);
        let conflicting_tenant = self
            .graph
            .node_ids_for_string_property_exact("slug", &tenant.slug)
            .into_iter()
            .filter(|candidate| candidate != &node_id)
            .filter_map(|candidate| self.graph.get_node(&candidate))
            .find_map(decode_native_tenant_record);
        if let Some(existing) = conflicting_tenant {
            return Err(GraphStoreError::new(
                "catalog_unique_violation",
                format!(
                    "tenant slug {} already belongs to {}",
                    tenant.slug, existing.tenant_id
                ),
            ));
        }
        self.upsert_record(
            node_id,
            NATIVE_TENANT_LABEL,
            &tenant,
            decode_native_tenant_record,
        )
    }

    pub fn upsert_project(&mut self, project: NativeProjectRecord) -> GraphStoreResult<()> {
        self.require_tenant(&project.tenant_id)?;
        require_non_empty(&project.project_slug, "project_slug")?;
        let node_id =
            native_catalog_node_id("project", &[&project.tenant_id, &project.project_slug]);
        self.upsert_record(
            node_id,
            NATIVE_PROJECT_LABEL,
            &project,
            decode_native_project_record,
        )
    }

    pub fn upsert_billing_account(
        &mut self,
        account: NativeBillingAccountRecord,
    ) -> GraphStoreResult<()> {
        self.require_tenant(&account.tenant_id)?;
        require_non_empty(&account.plan, "plan")?;
        require_non_empty(&account.status, "status")?;
        let node_id = native_catalog_node_id("billing", &[&account.tenant_id]);
        self.upsert_record(
            node_id,
            NATIVE_BILLING_LABEL,
            &account,
            decode_native_billing_record,
        )
    }

    pub fn upsert_auth_principal(
        &mut self,
        principal: NativeAuthPrincipalRecord,
    ) -> GraphStoreResult<()> {
        self.require_tenant(&principal.tenant_id)?;
        require_non_empty(&principal.principal_id, "principal_id")?;
        require_non_empty(&principal.kind, "kind")?;
        let node_id = native_catalog_node_id("principal", &[&principal.principal_id]);
        self.upsert_record(
            node_id,
            NATIVE_AUTH_PRINCIPAL_LABEL,
            &principal,
            decode_native_auth_principal_record,
        )
    }

    fn upsert_record<T: Serialize, D>(
        &mut self,
        node_id: String,
        label: &str,
        record: &T,
        decode: fn(&NodeRecord) -> Option<D>,
    ) -> GraphStoreResult<()> {
        if self
            .graph
            .get_node_record_including_tombstone(&node_id)
            .is_some_and(|existing| decode(&existing).is_none())
        {
            return Err(GraphStoreError::new(
                "catalog_reserved_id_collision",
                format!("reserved catalog node {node_id} is not a canonical {label} record"),
            ));
        }
        let properties = serde_json::to_value(record).map_err(|error| {
            GraphStoreError::new("catalog_record_encode_failed", error.to_string())
        })?;
        self.graph
            .upsert_node(NodeRecord::new(node_id, [label], properties))?;
        Ok(())
    }
}

fn native_catalog_node_id(kind: &str, components: &[&str]) -> String {
    let mut node_id = format!("{NATIVE_CATALOG_NODE_PREFIX}{kind}");
    for component in components {
        node_id.push(':');
        node_id.push_str(&component.len().to_string());
        node_id.push(':');
        node_id.push_str(component);
    }
    node_id
}

fn node_has_exact_label(node: &NodeRecord, label: &str) -> bool {
    node.labels.iter().any(|candidate| candidate == label)
}

fn decode_native_catalog_record<T: DeserializeOwned>(node: &NodeRecord) -> Option<T> {
    serde_json::from_value(node.properties.clone()).ok()
}

fn decode_native_tenant_record(node: &NodeRecord) -> Option<NativeTenantRecord> {
    if !node_has_native_catalog_family(node, NATIVE_TENANT_LABEL) {
        return None;
    }
    let record: NativeTenantRecord = decode_native_catalog_record(node)?;
    (has_text(&record.tenant_id)
        && has_text(&record.slug)
        && node.id == native_catalog_node_id("tenant", &[&record.tenant_id]))
    .then_some(record)
}

fn decode_native_project_record(node: &NodeRecord) -> Option<NativeProjectRecord> {
    if !node_has_native_catalog_family(node, NATIVE_PROJECT_LABEL) {
        return None;
    }
    let record: NativeProjectRecord = decode_native_catalog_record(node)?;
    (has_text(&record.tenant_id)
        && has_text(&record.project_slug)
        && node.id == native_catalog_node_id("project", &[&record.tenant_id, &record.project_slug]))
    .then_some(record)
}

fn decode_native_billing_record(node: &NodeRecord) -> Option<NativeBillingAccountRecord> {
    if !node_has_native_catalog_family(node, NATIVE_BILLING_LABEL) {
        return None;
    }
    let record: NativeBillingAccountRecord = decode_native_catalog_record(node)?;
    (has_text(&record.tenant_id)
        && has_text(&record.plan)
        && has_text(&record.status)
        && node.id == native_catalog_node_id("billing", &[&record.tenant_id]))
    .then_some(record)
}

fn decode_native_auth_principal_record(node: &NodeRecord) -> Option<NativeAuthPrincipalRecord> {
    if !node_has_native_catalog_family(node, NATIVE_AUTH_PRINCIPAL_LABEL) {
        return None;
    }
    let record: NativeAuthPrincipalRecord = decode_native_catalog_record(node)?;
    (has_text(&record.principal_id)
        && has_text(&record.tenant_id)
        && has_text(&record.kind)
        && node.id == native_catalog_node_id("principal", &[&record.principal_id]))
    .then_some(record)
}

fn node_has_native_catalog_family(node: &NodeRecord, expected_label: &str) -> bool {
    node_has_exact_label(node, expected_label)
        && node
            .labels
            .iter()
            .all(|label| !is_native_catalog_label(label) || label == expected_label)
}

fn is_native_catalog_label(label: &str) -> bool {
    matches!(
        label,
        NATIVE_TENANT_LABEL
            | NATIVE_PROJECT_LABEL
            | NATIVE_BILLING_LABEL
            | NATIVE_AUTH_PRINCIPAL_LABEL
    )
}

fn has_text(value: &str) -> bool {
    !value.trim().is_empty()
}

fn require_non_empty(value: &str, field: &str) -> GraphStoreResult<()> {
    if !has_text(value) {
        Err(GraphStoreError::new(
            "catalog_empty_field",
            format!("{field} must not be empty"),
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn camel_labels_preserve_acronym_runs() {
        assert_eq!(camel_label_to_snake("DatasetVersion"), "dataset_version");
        assert_eq!(camel_label_to_snake("HTTPServer"), "http_server");
        assert_eq!(
            camel_label_to_snake("OAuthHTTPServer2"),
            "o_auth_http_server2"
        );
        assert_eq!(camel_label_to_snake("already_snake"), "already_snake");
    }

    #[test]
    fn snake_label_alias_does_not_shadow_an_edge_relation() {
        let mut graph = crate::InMemoryGraphStore::default();
        graph
            .upsert_node(NodeRecord::new("version:1", ["DatasetVersion"], json!({})))
            .unwrap();
        graph
            .upsert_node(NodeRecord::new("version:2", ["Target"], json!({})))
            .unwrap();
        graph
            .upsert_edge(EdgeRecord::new(
                "edge:version",
                "version:1",
                "dataset_version",
                "version:2",
                json!({}),
            ))
            .unwrap();

        let definitions = discover_definitions(&graph);
        assert_eq!(
            definitions
                .get("dataset_version")
                .map(|definition| &definition.source),
            Some(&RelationSource::EdgeType("dataset_version".into()))
        );
        assert_eq!(
            definitions
                .get("datasetversion")
                .map(|definition| &definition.source),
            Some(&RelationSource::Label("datasetversion".into()))
        );
    }

    #[test]
    fn native_catalog_enforces_foreign_keys_and_uniqueness() {
        let mut catalog = NativeCatalog::new();
        let missing_tenant = catalog.upsert_project(NativeProjectRecord {
            tenant_id: "tenant:missing".to_string(),
            project_slug: "theorem".to_string(),
            display_name: None,
        });
        assert_eq!(
            missing_tenant.unwrap_err().code,
            "catalog_foreign_key_violation"
        );

        catalog
            .upsert_tenant(NativeTenantRecord {
                tenant_id: "tenant:a".to_string(),
                slug: "a".to_string(),
                display_name: Some("Tenant A".to_string()),
            })
            .unwrap();
        catalog
            .upsert_project(NativeProjectRecord {
                tenant_id: "tenant:a".to_string(),
                project_slug: "theorem".to_string(),
                display_name: Some("Theorem".to_string()),
            })
            .unwrap();
        catalog
            .upsert_billing_account(NativeBillingAccountRecord {
                tenant_id: "tenant:a".to_string(),
                plan: "pro".to_string(),
                status: "active".to_string(),
            })
            .unwrap();
        catalog
            .upsert_auth_principal(NativeAuthPrincipalRecord {
                principal_id: "principal:1".to_string(),
                tenant_id: "tenant:a".to_string(),
                kind: "api_key".to_string(),
                token_hash: Some("sha256:redacted".to_string()),
                scopes: vec!["memory:read".to_string()],
            })
            .unwrap();

        assert_eq!(catalog.projects_for_tenant("tenant:a").len(), 1);
        assert_eq!(catalog.auth_principals_for_tenant("tenant:a").len(), 1);

        let duplicate_slug = catalog.upsert_tenant(NativeTenantRecord {
            tenant_id: "tenant:b".to_string(),
            slug: "a".to_string(),
            display_name: None,
        });
        assert_eq!(duplicate_slug.unwrap_err().code, "catalog_unique_violation");
    }
}
