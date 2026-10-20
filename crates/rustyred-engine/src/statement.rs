use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::epistemic::{
    EpistemicSourceKind, EPISTEMIC_SHADOW_LABEL, HAS_EPISTEMIC_SHADOW, SAME_ECLASS,
};
use crate::graph::capability::GraphStore;
use crate::graph::error::GraphStoreResult;
use crate::graph::query::{NeighborQuery, NodeQuery};
use crate::graph::record::{EdgeRecord, EpistemicType, NodeRecord, Provenance};
use crate::state::stable_hash;

pub const STATEMENT_LABEL: &str = "Statement";
pub const PREDICATE_LABEL: &str = "Predicate";
pub const HAS_SUBJECT: &str = "HasSubject";
pub const HAS_OBJECT: &str = "HasObject";
pub const HAS_PREDICATE: &str = "HasPredicate";
pub const SAME_AS: &str = "SameAs";
pub const CANONICAL_ENTITY_LABEL: &str = "CanonicalEntity";

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
pub struct Confidence(pub f64);

impl Confidence {
    pub fn new(value: f64) -> Self {
        if value.is_finite() {
            Self(value.clamp(0.0, 1.0))
        } else {
            Self(0.0)
        }
    }

    pub fn value(self) -> f64 {
        self.0
    }

    pub fn join(a: Confidence, b: Confidence) -> Confidence {
        Confidence::new(a.0.max(b.0))
    }

    pub fn conjoin(premises: &[Confidence], decay: f64) -> Confidence {
        if premises.is_empty() {
            return Confidence::new(decay);
        }
        let min = premises
            .iter()
            .map(|confidence| confidence.0)
            .fold(1.0_f64, f64::min);
        Confidence::new(decay * min)
    }
}

impl From<f64> for Confidence {
    fn from(value: f64) -> Self {
        Confidence::new(value)
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "PascalCase")]
pub enum StatementSemiring {
    Boolean,
    Counting,
    #[default]
    Viterbi,
    Tropical,
    Probability,
    Corroboration,
    Why,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StatementFieldProvenance {
    pub source_kind: EpistemicSourceKind,
    pub engine: String,
    pub version: String,
    pub computed_at: i64,
}

impl StatementFieldProvenance {
    pub fn new(
        source_kind: EpistemicSourceKind,
        engine: impl Into<String>,
        version: impl Into<String>,
        computed_at: i64,
    ) -> Self {
        Self {
            source_kind,
            engine: engine.into(),
            version: version.into(),
            computed_at,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StatementProvenance {
    pub field_provenance: StatementFieldProvenance,
    pub dependency_fact_ids: Vec<String>,
    pub rule_id: String,
    pub semiring: StatementSemiring,
}

impl StatementProvenance {
    pub fn new(
        field_provenance: StatementFieldProvenance,
        dependency_fact_ids: impl IntoIterator<Item = impl Into<String>>,
        rule_id: impl Into<String>,
        semiring: StatementSemiring,
    ) -> Self {
        Self {
            field_provenance,
            dependency_fact_ids: sorted_strings(dependency_fact_ids),
            rule_id: rule_id.into(),
            semiring,
        }
    }

    pub fn why(&self) -> Vec<String> {
        sorted_strings(self.dependency_fact_ids.clone())
    }
}

pub struct StatementRecord;

impl StatementRecord {
    pub fn assert(
        subject_ref: impl AsRef<str>,
        predicate_key: impl AsRef<str>,
        object_ref: impl AsRef<str>,
        props_json: Value,
    ) -> GraphStoreResult<NodeRecord> {
        statement_node(
            subject_ref.as_ref(),
            predicate_key.as_ref(),
            object_ref.as_ref(),
            props_json,
            StatementKind::Asserted,
            None,
        )
    }

    pub fn assert_literal(
        subject_ref: impl AsRef<str>,
        predicate_key: impl AsRef<str>,
        object_value: Value,
        props_json: Value,
    ) -> GraphStoreResult<NodeRecord> {
        let mut properties = object_map(props_json);
        properties.insert("object_value".to_string(), object_value.clone());
        let object_ref = literal_ref(&object_value)?;
        Self::assert(
            subject_ref,
            predicate_key,
            object_ref,
            Value::Object(properties),
        )
    }

    pub fn derive(
        subject_ref: impl AsRef<str>,
        predicate_key: impl AsRef<str>,
        object_ref: impl AsRef<str>,
        provenance: StatementProvenance,
        props_json: Value,
    ) -> GraphStoreResult<NodeRecord> {
        statement_node(
            subject_ref.as_ref(),
            predicate_key.as_ref(),
            object_ref.as_ref(),
            props_json,
            StatementKind::Derived,
            Some(provenance),
        )
    }

    pub fn derive_literal(
        subject_ref: impl AsRef<str>,
        predicate_key: impl AsRef<str>,
        object_value: Value,
        provenance: StatementProvenance,
        props_json: Value,
    ) -> GraphStoreResult<NodeRecord> {
        let mut properties = object_map(props_json);
        properties.insert("object_value".to_string(), object_value.clone());
        let object_ref = literal_ref(&object_value)?;
        Self::derive(
            subject_ref,
            predicate_key,
            object_ref,
            provenance,
            Value::Object(properties),
        )
    }

    pub fn incidence_edges(statement: &NodeRecord) -> GraphStoreResult<Vec<EdgeRecord>> {
        statement_incidence_edges(statement)
    }

    pub fn dependency_fact_ids(statement: &NodeRecord) -> Vec<String> {
        statement
            .properties
            .get("provenance")
            .and_then(|value| value.get("dependency_fact_ids"))
            .or_else(|| statement.properties.get("dependency_fact_ids"))
            .map(value_string_vec)
            .unwrap_or_default()
    }

    pub fn provenance(statement: &NodeRecord) -> Option<StatementProvenance> {
        statement
            .properties
            .get("provenance")
            .cloned()
            .and_then(|value| serde_json::from_value(value).ok())
    }

    pub fn why(statement: &NodeRecord) -> Vec<String> {
        Self::provenance(statement)
            .map(|provenance| provenance.why())
            .unwrap_or_else(|| Self::dependency_fact_ids(statement))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StatementKind {
    Asserted,
    Derived,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StatementWriteReceipt {
    pub statement_id: String,
    pub edge_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StatementQuery {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_ref: Option<String>,
    #[serde(default = "default_true")]
    pub include_asserted: bool,
    #[serde(default = "default_true")]
    pub include_derived: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<usize>,
}

impl Default for StatementQuery {
    fn default() -> Self {
        Self {
            subject_id: None,
            relation: None,
            object_ref: None,
            include_asserted: true,
            include_derived: true,
            limit: None,
        }
    }
}

impl StatementQuery {
    pub fn subject(subject_id: impl Into<String>) -> Self {
        Self {
            subject_id: Some(subject_id.into()),
            ..Self::default()
        }
    }

    pub fn with_relation(mut self, relation: impl Into<String>) -> Self {
        self.relation = Some(relation.into());
        self
    }

    pub fn with_object_ref(mut self, object_ref: impl Into<String>) -> Self {
        self.object_ref = Some(object_ref.into());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        if limit > 0 {
            self.limit = Some(limit);
        }
        self
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum FlatObject {
    Entity(String),
    Literal(Value),
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FlatTriple {
    pub statement_id: String,
    pub subject_id: String,
    pub relation: String,
    pub object: FlatObject,
    pub confidence: Confidence,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<StatementProvenance>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance_handle: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct EpistemicShadowStatementMigrationReport {
    pub statements_written: usize,
    pub edges_written: usize,
    pub skipped: usize,
    pub shadow_ids: Vec<String>,
}

pub fn statement_id(
    subject_ref: &str,
    predicate_key: &str,
    object_ref: &str,
) -> GraphStoreResult<String> {
    Ok(format!(
        "statement:{}",
        stable_hash(json!([subject_ref, predicate_key, object_ref]))?
    ))
}

pub fn literal_ref(value: &Value) -> GraphStoreResult<String> {
    Ok(format!("lit:{}", stable_hash(value)?))
}

pub fn predicate_id(predicate_key: &str) -> String {
    format!("predicate:{}", predicate_key)
}

pub fn statement_incidence_edge_id(
    statement_id: &str,
    edge_type: &str,
    target_ref: &str,
) -> GraphStoreResult<String> {
    Ok(format!(
        "statement:incidence:{}",
        stable_hash(json!([statement_id, edge_type, target_ref]))?
    ))
}

pub fn predicate_node(predicate_key: &str) -> NodeRecord {
    NodeRecord::new(
        predicate_id(predicate_key),
        [PREDICATE_LABEL],
        json!({ "predicate_key": predicate_key }),
    )
}

pub fn statement_incidence_edges(statement: &NodeRecord) -> GraphStoreResult<Vec<EdgeRecord>> {
    let mut edges = Vec::new();
    let statement_id = statement.id.as_str();
    let subject_ref = prop_str(&statement.properties, "subject_ref")
        .or_else(|| prop_str(&statement.properties, "subject_id"));
    let object_ref = prop_str(&statement.properties, "object_ref")
        .or_else(|| prop_str(&statement.properties, "object_id"));
    let confidence = statement_confidence(statement).value();
    let provenance = statement_edge_provenance(statement);
    let edge_properties = statement_edge_properties(statement);

    if let Some(subject_ref) = subject_ref {
        let mut edge = EdgeRecord::new(
            statement_incidence_edge_id(statement_id, HAS_SUBJECT, &subject_ref)?,
            statement_id,
            HAS_SUBJECT,
            &subject_ref,
            edge_properties.clone(),
        )
        .with_confidence(confidence)
        .with_epistemic_type(EpistemicType::Derives);
        if let Some(provenance) = provenance.clone() {
            edge = edge.with_provenance(provenance);
        }
        edges.push(edge);
    }

    if let Some(object_ref) = object_ref.filter(|value| !is_literal_ref(value)) {
        let mut edge = EdgeRecord::new(
            statement_incidence_edge_id(statement_id, HAS_OBJECT, &object_ref)?,
            statement_id,
            HAS_OBJECT,
            &object_ref,
            edge_properties,
        )
        .with_confidence(confidence)
        .with_epistemic_type(EpistemicType::Derives);
        if let Some(provenance) = provenance {
            edge = edge.with_provenance(provenance);
        }
        edges.push(edge);
    }

    Ok(edges)
}

pub fn predicate_incidence_edge(statement: &NodeRecord) -> GraphStoreResult<Option<EdgeRecord>> {
    let relation = prop_str(&statement.properties, "relation")
        .or_else(|| prop_str(&statement.properties, "predicate_key"));
    let Some(relation) = relation else {
        return Ok(None);
    };
    let predicate_id = predicate_id(&relation);
    let confidence = statement_confidence(statement).value();
    let mut edge = EdgeRecord::new(
        statement_incidence_edge_id(&statement.id, HAS_PREDICATE, &predicate_id)?,
        &statement.id,
        HAS_PREDICATE,
        &predicate_id,
        statement_edge_properties(statement),
    )
    .with_confidence(confidence)
    .with_epistemic_type(EpistemicType::Derives);
    if let Some(provenance) = statement_edge_provenance(statement) {
        edge = edge.with_provenance(provenance);
    }
    Ok(Some(edge))
}

pub fn write_statement<S: GraphStore>(
    store: &mut S,
    statement: NodeRecord,
    promote_predicate: bool,
) -> GraphStoreResult<StatementWriteReceipt> {
    let statement_id = statement.id.clone();
    let relation = prop_str(&statement.properties, "relation");
    store.upsert_node(statement.clone())?;

    let mut edge_ids = Vec::new();
    for edge in statement_incidence_edges(&statement)? {
        let id = edge.id.clone();
        store.upsert_edge(edge)?;
        edge_ids.push(id);
    }

    let predicate_id = if promote_predicate {
        if let Some(relation) = relation {
            let predicate = predicate_node(&relation);
            let id = predicate.id.clone();
            store.upsert_node(predicate)?;
            if let Some(edge) = predicate_incidence_edge(&statement)? {
                let edge_id = edge.id.clone();
                store.upsert_edge(edge)?;
                edge_ids.push(edge_id);
            }
            Some(id)
        } else {
            None
        }
    } else {
        None
    };

    Ok(StatementWriteReceipt {
        statement_id,
        edge_ids,
        predicate_id,
    })
}

pub fn promote_statement_predicate<S: GraphStore>(
    store: &mut S,
    statement_id: &str,
) -> GraphStoreResult<Option<EdgeRecord>> {
    let Some(statement) = store.get_node(statement_id).cloned() else {
        return Ok(None);
    };
    let Some(relation) = prop_str(&statement.properties, "relation") else {
        return Ok(None);
    };
    store.upsert_node(predicate_node(&relation))?;
    let Some(edge) = predicate_incidence_edge(&statement)? else {
        return Ok(None);
    };
    store.upsert_edge(edge.clone())?;
    Ok(Some(edge))
}

pub fn flatten_statements<S: GraphStore>(store: &S, query: StatementQuery) -> Vec<FlatTriple> {
    let mut statements = store
        .query_nodes(NodeQuery::label(STATEMENT_LABEL).with_limit(query.limit.unwrap_or(100_000)));
    statements.sort_by(|left, right| left.id.cmp(&right.id));

    let mut triples = Vec::new();
    for statement in statements {
        let asserted = statement
            .properties
            .get("asserted")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let derived = statement
            .properties
            .get("derived")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if asserted && !query.include_asserted {
            continue;
        }
        if derived && !query.include_derived {
            continue;
        }

        let subject_id = incidence_target(store, &statement.id, HAS_SUBJECT)
            .or_else(|| prop_str(&statement.properties, "subject_ref"))
            .or_else(|| prop_str(&statement.properties, "subject_id"));
        let Some(subject_id) = subject_id else {
            continue;
        };
        if query
            .subject_id
            .as_deref()
            .is_some_and(|id| id != subject_id)
        {
            continue;
        }

        let relation = prop_str(&statement.properties, "relation")
            .or_else(|| prop_str(&statement.properties, "predicate_key"));
        let Some(relation) = relation else {
            continue;
        };
        if query
            .relation
            .as_deref()
            .is_some_and(|wanted| wanted != relation)
        {
            continue;
        }

        let object_ref = prop_str(&statement.properties, "object_ref")
            .or_else(|| prop_str(&statement.properties, "object_id"));
        if query
            .object_ref
            .as_deref()
            .is_some_and(|wanted| object_ref.as_deref() != Some(wanted))
        {
            continue;
        }

        let object = if let Some(object_id) = incidence_target(store, &statement.id, HAS_OBJECT) {
            FlatObject::Entity(object_id)
        } else if let Some(value) = statement.properties.get("object_value").cloned() {
            FlatObject::Literal(value)
        } else if let Some(object_ref) = object_ref {
            if is_literal_ref(&object_ref) {
                FlatObject::Literal(Value::String(object_ref))
            } else {
                FlatObject::Entity(object_ref)
            }
        } else {
            continue;
        };

        triples.push(FlatTriple {
            statement_id: statement.id.clone(),
            subject_id,
            relation,
            object,
            confidence: statement_confidence(&statement),
            provenance: StatementRecord::provenance(&statement),
            provenance_handle: prop_str(&statement.properties, "fact_id")
                .or_else(|| prop_str(&statement.properties, "source_shadow_id"))
                .or_else(|| prop_str(&statement.properties, "provenance_handle")),
        });
    }
    triples
}

pub fn propose_same_as<S: GraphStore>(
    store: &mut S,
    a_id: &str,
    b_id: &str,
    confidence: Confidence,
    dependency_fact_ids: impl IntoIterator<Item = impl Into<String>>,
) -> GraphStoreResult<EdgeRecord> {
    let (from_id, to_id) = canonical_pair(a_id, b_id);
    let dependency_fact_ids = sorted_strings(dependency_fact_ids);
    let edge = EdgeRecord::new(
        same_as_edge_id(&from_id, &to_id)?,
        from_id.clone(),
        SAME_AS,
        to_id.clone(),
        json!({
            "confidence": confidence.value(),
            "dependency_fact_ids": dependency_fact_ids,
            "quarantine": false,
        }),
    )
    .with_confidence(confidence.value())
    .with_epistemic_type(EpistemicType::Derives)
    .with_provenance(Provenance {
        source_id: Some("statement.same_as".to_string()),
        timestamp: None,
        method: Some("confidence_weighted_same_as".to_string()),
    });
    store.upsert_edge(edge.clone())?;
    Ok(edge)
}

pub fn collapse_if_corroborated<S: GraphStore>(
    store: &mut S,
    cluster: &[String],
    threshold: Confidence,
) -> GraphStoreResult<Option<NodeRecord>> {
    let mut members = sorted_strings(cluster.iter().cloned());
    members.retain(|id| store.get_node(id).is_some());
    if members.len() < 2 {
        return Ok(None);
    }

    let member_set = members.iter().cloned().collect::<BTreeSet<_>>();
    let mut best_confidence = 0.0_f64;
    let mut dependency_fact_ids = BTreeSet::new();
    for member in &members {
        for hit in store
            .neighbors(NeighborQuery::out(member).with_edge_type(SAME_AS))
            .into_iter()
            .chain(store.neighbors(NeighborQuery::in_(member).with_edge_type(SAME_AS)))
        {
            let Some(edge) = store.get_edge(&hit.edge_id) else {
                continue;
            };
            if !member_set.contains(&edge.from_id) || !member_set.contains(&edge.to_id) {
                continue;
            }
            best_confidence = best_confidence.max(edge.effective_confidence());
            dependency_fact_ids.extend(prop_string_vec(&edge.properties, "dependency_fact_ids"));
        }
    }

    let independent_sources = dependency_fact_ids.iter().cloned().collect::<BTreeSet<_>>();
    if best_confidence < threshold.value() || independent_sources.len() < 2 {
        return Ok(None);
    }

    let canonical_id = canonical_entity_id(&members)?;
    let canonical = NodeRecord::new(
        canonical_id.clone(),
        ["Entity", CANONICAL_ENTITY_LABEL],
        json!({
            "canonical_for": members,
            "same_as_confidence": best_confidence,
            "dependency_fact_ids": dependency_fact_ids.iter().cloned().collect::<Vec<_>>(),
            "independent_source_count": independent_sources.len(),
            "collapse_threshold": threshold.value(),
            "versioned_base_record": true,
        }),
    );
    store.upsert_node(canonical.clone())?;
    for member in &members {
        let edge = EdgeRecord::new(
            same_eclass_collapse_edge_id(member, &canonical_id)?,
            member,
            SAME_ECLASS,
            &canonical_id,
            json!({
                "class_id": canonical_id,
                "canonical_form": canonical_id,
                "confidence": best_confidence,
                "dependency_fact_ids": dependency_fact_ids.iter().cloned().collect::<Vec<_>>(),
                "evidence": "same_as_corroborated_collapse",
                "source_kind": "structural",
                "versioned_base_record": true,
            }),
        )
        .with_confidence(best_confidence)
        .with_provenance(Provenance {
            source_id: Some("statement.same_as".to_string()),
            timestamp: None,
            method: Some("same_as_corroborated_collapse".to_string()),
        });
        store.upsert_edge(edge)?;
    }
    Ok(Some(canonical))
}

pub fn migrate_epistemic_shadows_to_statements<S: GraphStore>(
    store: &mut S,
    engine: impl AsRef<str>,
    version: impl AsRef<str>,
    computed_at: i64,
) -> GraphStoreResult<EpistemicShadowStatementMigrationReport> {
    let shadows = store.query_nodes(NodeQuery::label(EPISTEMIC_SHADOW_LABEL).with_limit(100_000));
    let mut report = EpistemicShadowStatementMigrationReport::default();
    for shadow in shadows {
        let content_id = prop_str(&shadow.properties, "content_node_id");
        let Some(content_id) = content_id else {
            report.skipped += 1;
            continue;
        };
        if store.get_node(&content_id).is_none() {
            report.skipped += 1;
            continue;
        }

        let provenance = StatementProvenance::new(
            StatementFieldProvenance::new(
                EpistemicSourceKind::Structural,
                engine.as_ref(),
                version.as_ref(),
                computed_at,
            ),
            [shadow.id.clone()],
            "epistemic_shadow_statement_migration",
            StatementSemiring::Why,
        );
        let statement = StatementRecord::derive(
            &content_id,
            "epistemic_shadow_claim",
            &shadow.id,
            provenance,
            json!({
                "confidence": shadow
                    .properties
                    .get("confidence")
                    .and_then(Value::as_f64)
                    .unwrap_or(1.0),
                "source_shadow_id": shadow.id,
                "grounded_extension_status": prop_str(&shadow.properties, "grounded_extension_status")
                    .unwrap_or_else(|| "undecided".to_string()),
                "quarantine": true,
            }),
        )?;
        let receipt = write_statement(store, statement, false)?;
        ensure_shadow_edge(store, &content_id, &shadow.id)?;
        report.statements_written += 1;
        report.edges_written += receipt.edge_ids.len();
        report.shadow_ids.push(shadow.id);
    }
    report.shadow_ids.sort();
    report.shadow_ids.dedup();
    Ok(report)
}

pub fn canonical_entity_id(members: &[String]) -> GraphStoreResult<String> {
    Ok(format!("entity:canonical:{}", stable_hash(members)?))
}

pub fn same_as_edge_id(a_id: &str, b_id: &str) -> GraphStoreResult<String> {
    let (from_id, to_id) = canonical_pair(a_id, b_id);
    Ok(format!("same_as:{}", stable_hash(json!([from_id, to_id]))?))
}

fn statement_node(
    subject_ref: &str,
    predicate_key: &str,
    object_ref: &str,
    props_json: Value,
    kind: StatementKind,
    provenance: Option<StatementProvenance>,
) -> GraphStoreResult<NodeRecord> {
    let mut properties = object_map(props_json);
    let confidence = properties
        .get("confidence")
        .and_then(Value::as_f64)
        .map(Confidence::new)
        .unwrap_or(Confidence(1.0));
    properties.insert("subject_ref".to_string(), json!(subject_ref));
    properties.insert("subject_id".to_string(), json!(subject_ref));
    properties.insert("predicate_key".to_string(), json!(predicate_key));
    properties.insert("relation".to_string(), json!(predicate_key));
    properties.insert("object_ref".to_string(), json!(object_ref));
    properties.insert("confidence".to_string(), json!(confidence.value()));
    match kind {
        StatementKind::Asserted => {
            properties.insert("asserted".to_string(), json!(true));
            properties.insert("derived".to_string(), json!(false));
        }
        StatementKind::Derived => {
            properties.insert("asserted".to_string(), json!(false));
            properties.insert("derived".to_string(), json!(true));
            properties.insert("quarantine".to_string(), json!(true));
        }
    }
    if let Some(provenance) = provenance {
        if !properties.contains_key("dependency_fact_ids") {
            properties.insert(
                "dependency_fact_ids".to_string(),
                json!(provenance.dependency_fact_ids.clone()),
            );
        }
        if !properties.contains_key("rule_id") {
            properties.insert("rule_id".to_string(), json!(provenance.rule_id.clone()));
        }
        properties.insert(
            "semiring".to_string(),
            serde_json::to_value(&provenance.semiring).unwrap_or_else(|_| json!("Why")),
        );
        properties.insert(
            "provenance".to_string(),
            serde_json::to_value(provenance).unwrap_or_else(|_| json!({})),
        );
    }

    Ok(NodeRecord::new(
        statement_id(subject_ref, predicate_key, object_ref)?,
        [STATEMENT_LABEL],
        Value::Object(properties),
    ))
}

fn statement_confidence(statement: &NodeRecord) -> Confidence {
    statement
        .properties
        .get("confidence")
        .and_then(Value::as_f64)
        .map(Confidence::new)
        .unwrap_or(Confidence(1.0))
}

fn statement_edge_properties(statement: &NodeRecord) -> Value {
    let mut properties = Map::new();
    for key in [
        "relation",
        "predicate_key",
        "fact_id",
        "rule_id",
        "confidence",
        "dependency_fact_ids",
        "engine",
        "engine_version",
        "computed_at",
        "component_key",
        "quarantine",
    ] {
        if let Some(value) = statement.properties.get(key).cloned() {
            properties.insert(key.to_string(), value);
        }
    }
    Value::Object(properties)
}

fn statement_edge_provenance(statement: &NodeRecord) -> Option<Provenance> {
    let provenance = StatementRecord::provenance(statement)?;
    Some(Provenance {
        source_id: Some(provenance.field_provenance.engine),
        timestamp: Some(provenance.field_provenance.computed_at.to_string()),
        method: Some(provenance.rule_id),
    })
}

fn incidence_target<S: GraphStore>(
    store: &S,
    statement_id: &str,
    edge_type: &str,
) -> Option<String> {
    store
        .neighbors(NeighborQuery::out(statement_id).with_edge_type(edge_type))
        .into_iter()
        .map(|hit| hit.node_id)
        .next()
}

fn ensure_shadow_edge<S: GraphStore>(
    store: &mut S,
    content_id: &str,
    shadow_id: &str,
) -> GraphStoreResult<()> {
    if store
        .neighbors(NeighborQuery::out(content_id).with_edge_type(HAS_EPISTEMIC_SHADOW))
        .into_iter()
        .any(|hit| hit.node_id == shadow_id)
    {
        return Ok(());
    }
    store.upsert_edge(EdgeRecord::new(
        format!(
            "has-epistemic-shadow:{}",
            stable_hash(json!([content_id, shadow_id]))?
        ),
        content_id,
        HAS_EPISTEMIC_SHADOW,
        shadow_id,
        json!({}),
    ))?;
    Ok(())
}

fn canonical_pair(a_id: &str, b_id: &str) -> (String, String) {
    if a_id <= b_id {
        (a_id.to_string(), b_id.to_string())
    } else {
        (b_id.to_string(), a_id.to_string())
    }
}

fn same_eclass_collapse_edge_id(member_id: &str, canonical_id: &str) -> GraphStoreResult<String> {
    Ok(format!(
        "same_eclass:{}",
        stable_hash(json!([member_id, canonical_id, "same_as_collapse"]))?
    ))
}

fn is_literal_ref(value: &str) -> bool {
    value.starts_with("lit:")
}

fn object_map(value: Value) -> Map<String, Value> {
    value.as_object().cloned().unwrap_or_default()
}

fn sorted_strings(values: impl IntoIterator<Item = impl Into<String>>) -> Vec<String> {
    let mut values = values
        .into_iter()
        .map(Into::into)
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

fn prop_string_vec(value: &Value, key: &str) -> Vec<String> {
    value.get(key).map(value_string_vec).unwrap_or_default()
}

fn value_string_vec(value: &Value) -> Vec<String> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str().map(str::to_string))
        .collect()
}

fn prop_str(value: &Value, key: &str) -> Option<String> {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .map(value_to_string)
        .filter(|value| !value.trim().is_empty())
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Number(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn default_true() -> bool {
    true
}
