//! A deterministic consumer of pinned graph/history projections (Wiki W1).
//! No writer, clock, network, alias nomination, or additional sealed FoldId.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use theorem_surface_contracts::article::*;

use crate::graph::record::{Direction, EdgeRecord, NodeRecord};
use crate::statement::{canonical_entity_id, StatementRecord};

pub mod links;
pub use links::fold_article_with_links;
pub mod snapshot;
pub mod source;

/// D1 has a closed relationship policy. These are persisted graph spellings,
/// compared case-insensitively; arbitrary new edge types are not traversed.
pub const ARTICLE_EDGE_KINDS: &[&str] = &[
    "memory_relates",
    "memory_in_project",
    "memory_cites",
    "memory_derived_from",
    "memory_supports",
    "memory_contradicts",
    "memory_undercuts",
    "memory_tension",
    "memory_supersedes",
    "memory_summarizes",
    "relates_to",
    "references",
    "cites",
    "supports",
    "contradicts",
    "derived_from",
    "derives",
    "consumes",
    "produces",
    "has_spec",
    "has_plan",
    "has_decision",
    "discusses",
    "evidence_for",
    "owned_by",
    "owner",
    "member_of",
    "works_on",
    "in_project",
    "depends_on",
];

const MAX_CLUSTER_READS: usize = 10_000;
const MAX_ALIAS_DEPTH: usize = 32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArticleTier {
    Gist,
    Overview,
    Full,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArticleFoldError {
    Missing(String),
    Excluded,
    AmbiguousIdentity(String),
    InvalidAuthority(String),
    IncompleteRead(String),
    Read(String),
}

impl std::fmt::Display for ArticleFoldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ArticleFoldError {}

#[derive(Clone, Debug)]
pub enum ArticleNode {
    Present(NodeRecord),
    Missing,
    /// A present but foreign, revoked, or out-of-scope record is never a gap.
    /// This variant intentionally carries no private record or identity.
    Excluded,
}

#[derive(Clone, Debug, Default)]
pub struct IdentityCandidates {
    pub nodes: Vec<NodeRecord>,
    pub excluded: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceAdmission {
    Local { dependency_node_ids: Vec<String> },
    Foreign { dependency_node_ids: Vec<String> },
    Unresolved,
}

#[derive(Clone, Debug)]
pub struct CitedEvidence {
    pub reference: Reference,
    /// Actual receipt, inputs, and source nodes whose changes affect replay.
    pub dependency_node_ids: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PinnedSection {
    pub section_id: String,
    pub block: BlockRef,
    /// Document record and snapshot artifact ids, resolved at this read pin.
    pub dependency_node_ids: Vec<String>,
}

/// Implementations bind one immutable graph root and current tenant/project
/// admission. Return errors for corruption or incomplete enumeration, not an
/// empty result. No method may mint records or perform a current-time lookup.
pub trait ArticleRead {
    fn node(&self, id: &str) -> Result<ArticleNode, ArticleFoldError>;
    /// Complete adjacency, including edges whose other endpoint is absent.
    /// Edges disallowed by present admission must not be returned.
    fn edges(&self, id: &str, direction: Direction) -> Result<Vec<EdgeRecord>, ArticleFoldError>;
    /// Exact id/slug, MemoryDocument.doc_id, MemoryNode.node_id,
    /// MemoryProject.project_slug, or persisted AliasWitness.alias. No fuzzy
    /// match, title guessing, room alias, or independently hashed canonical id.
    fn identity_candidates(&self, exact: &str) -> Result<IdentityCandidates, ArticleFoldError>;
    /// Join an exact source reference to persisted verified federation
    /// admission. Origin-scoped block URLs are not interchangeable with ids.
    fn source_admission(&self, source_ref: &str) -> Result<SourceAdmission, ArticleFoldError>;
    fn evidence(&self, node: &NodeRecord) -> Result<Vec<CitedEvidence>, ArticleFoldError>;
    fn pinned_sections(&self, canonical_id: &str) -> Result<Vec<PinnedSection>, ArticleFoldError>;
    /// Newest genuine commit intersecting these keys, under the captured
    /// head. Legacy absence is represented in Revision, never fabricated.
    fn revision(&self, dependency_keys: &BTreeSet<String>) -> Result<Revision, ArticleFoldError>;
}

#[derive(Default)]
struct Reads {
    ids: BTreeSet<String>,
    keys: BTreeSet<String>,
    admitted: BTreeMap<String, bool>,
    checking: BTreeSet<String>,
    count: usize,
}

impl Reads {
    fn add(&mut self, kind: &str, id: &str) -> Result<(), ArticleFoldError> {
        self.ids.insert(id.to_owned());
        self.keys.insert(format!("{kind}/{id}"));
        self.count += 1;
        if self.count > MAX_CLUSTER_READS {
            return Err(ArticleFoldError::IncompleteRead(
                "article read budget exceeded".into(),
            ));
        }
        Ok(())
    }
}

fn string<'a>(node: &'a NodeRecord, key: &str) -> Option<&'a str> {
    node.properties
        .get(key)
        .or_else(|| {
            if label(node, "CanonicalFact") {
                node.properties
                    .get("fact")
                    .and_then(|fact| fact.get("properties"))
                    .and_then(|properties| properties.get(key))
            } else {
                None
            }
        })
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}
fn strings(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}
fn label(node: &NodeRecord, name: &str) -> bool {
    node.labels.iter().any(|s| s == name)
}
fn edge_allowed(edge: &EdgeRecord) -> bool {
    !edge.tombstone && ARTICLE_EDGE_KINDS.contains(&edge.edge_type.to_ascii_lowercase().as_str())
}

fn source_refs(node: &NodeRecord) -> BTreeSet<String> {
    let mut refs = BTreeSet::new();
    for object in [
        Some(&node.properties),
        node.properties.get("metadata"),
        node.properties.get("provenance"),
        node.properties.get("fact"),
    ] {
        if let Some(object) = object {
            for key in ["source_ref", "source_url", "block_url"] {
                if let Some(value) = object
                    .get(key)
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                {
                    refs.insert(value.to_owned());
                }
            }
            refs.extend(strings(object.get("source_refs")));
        }
    }
    refs
}

/// Provenance closure affects admission but does not enlarge the article's
/// one-hop content neighborhood. Statement dependencies are the existing
/// StatementRecord authority, not an invented `quarantined` property.
fn admitted<R: ArticleRead + ?Sized>(
    read: &R,
    node: &NodeRecord,
    reads: &mut Reads,
) -> Result<bool, ArticleFoldError> {
    if let Some(value) = reads.admitted.get(&node.id) {
        return Ok(*value);
    }
    if label(node, "FederatedForeignCache") {
        reads.admitted.insert(node.id.clone(), false);
        return Ok(false);
    }
    if !reads.checking.insert(node.id.clone()) {
        return Err(ArticleFoldError::InvalidAuthority(
            "cyclic provenance dependency".into(),
        ));
    }
    let mut allowed = true;
    for source in source_refs(node) {
        match read.source_admission(&source)? {
            SourceAdmission::Local {
                dependency_node_ids,
            } => {
                for id in dependency_node_ids {
                    reads.add("node", &id)?;
                }
            }
            SourceAdmission::Foreign {
                dependency_node_ids,
            } => {
                for id in dependency_node_ids {
                    reads.add("node", &id)?;
                }
                allowed = false;
            }
            SourceAdmission::Unresolved => (),
        }
    }
    let mut dependencies = StatementRecord::dependency_fact_ids(node);
    if label(node, "Derivation") {
        dependencies.extend(strings(node.properties.get("input_refs")));
    }
    match read.evidence(node) {
        Ok(evidence) => {
            for receipt in evidence {
                dependencies.extend(
                    receipt
                        .dependency_node_ids
                        .into_iter()
                        .filter(|id| id != &node.id),
                );
            }
        }
        Err(ArticleFoldError::Excluded) => allowed = false,
        Err(error) => return Err(error),
    }
    dependencies.sort();
    dependencies.dedup();
    for dependency in dependencies {
        reads.add("node", &dependency)?;
        match read.node(&dependency)? {
            ArticleNode::Excluded => allowed = false,
            ArticleNode::Present(source) => allowed &= admitted(read, &source, reads)?,
            ArticleNode::Missing => {
                // An unresolved provenance input is not evidence of local
                // origin. Refuse projection until the authority can resolve it.
                return Err(ArticleFoldError::IncompleteRead(
                    "missing provenance dependency".into(),
                ));
            }
        }
    }
    reads.checking.remove(&node.id);
    reads.admitted.insert(node.id.clone(), allowed);
    Ok(allowed)
}

fn load<R: ArticleRead + ?Sized>(
    read: &R,
    id: &str,
    reads: &mut Reads,
) -> Result<ArticleNode, ArticleFoldError> {
    reads.add("node", id)?;
    match read.node(id)? {
        ArticleNode::Present(node) if !node.tombstone => {
            if admitted(read, &node, reads)? {
                Ok(ArticleNode::Present(node))
            } else {
                Ok(ArticleNode::Excluded)
            }
        }
        ArticleNode::Present(_) => Ok(ArticleNode::Missing),
        other => Ok(other),
    }
}

/// A persisted corroborated collapse changes the source record's identity. Scoped
/// readers must refuse its excluded target instead of silently dropping the
/// edge and restoring an obsolete member identity. Inbound unrelated aliases
/// remain ordinary filtered adjacency.
pub fn is_persisted_collapse_edge(edge: &EdgeRecord) -> bool {
    !edge.tombstone
        && (edge.edge_type == "SameEClass"
            && edge.properties.get("evidence").and_then(Value::as_str)
                == Some("same_as_corroborated_collapse")
            && edge
                .properties
                .get("versioned_base_record")
                .and_then(Value::as_bool)
                == Some(true))
}

fn canonicalize<R: ArticleRead + ?Sized>(
    read: &R,
    mut node: NodeRecord,
    reads: &mut Reads,
) -> Result<NodeRecord, ArticleFoldError> {
    let mut seen = BTreeSet::new();
    for _ in 0..MAX_ALIAS_DEPTH {
        if !seen.insert(node.id.clone()) {
            return Err(ArticleFoldError::InvalidAuthority(
                "canonical alias cycle".into(),
            ));
        }
        let mut targets = BTreeMap::new();
        for edge in read.edges(&node.id, Direction::Out)? {
            let is_collapse = edge.edge_type == "SameEClass" && is_persisted_collapse_edge(&edge);
            let is_witness = label(&node, "AliasWitness")
                && edge.edge_type == "ALIAS_WITNESS_FOR"
                && string(&node, "authority") == Some("witness_not_truth");
            if edge.tombstone || (!is_collapse && !is_witness) {
                continue;
            }
            reads.add("edge", &edge.id)?;
            let target = match load(read, &edge.to_id, reads)? {
                ArticleNode::Present(target) => target,
                ArticleNode::Excluded => return Err(ArticleFoldError::Excluded),
                ArticleNode::Missing => {
                    return Err(ArticleFoldError::InvalidAuthority(
                        "canonical target is absent".into(),
                    ));
                }
            };
            if is_collapse {
                let mut members = strings(target.properties.get("canonical_for"));
                members.sort();
                members.dedup();
                let mut facts = strings(target.properties.get("dependency_fact_ids"));
                facts.sort();
                facts.dedup();
                let confidence = target
                    .properties
                    .get("same_as_confidence")
                    .and_then(Value::as_f64);
                let threshold = target
                    .properties
                    .get("collapse_threshold")
                    .and_then(Value::as_f64);
                let valid = label(&target, "CanonicalEntity")
                    && target
                        .properties
                        .get("versioned_base_record")
                        .and_then(Value::as_bool)
                        == Some(true)
                    && target
                        .properties
                        .get("independent_source_count")
                        .and_then(Value::as_u64)
                        == Some(facts.len() as u64)
                    && edge.properties.get("class_id").and_then(Value::as_str)
                        == Some(target.id.as_str())
                    && edge
                        .properties
                        .get("canonical_form")
                        .and_then(Value::as_str)
                        == Some(target.id.as_str())
                    && members.len() >= 2
                    && members.contains(&node.id)
                    && facts.len() >= 2
                    && confidence.zip(threshold).is_some_and(|(c, t)| {
                        c.is_finite() && t.is_finite() && (0.0..=1.0).contains(&t) && c >= t
                    })
                    && canonical_entity_id(&members)
                        .map_err(|e| ArticleFoldError::Read(e.to_string()))?
                        == target.id;
                if !valid {
                    return Err(ArticleFoldError::InvalidAuthority(
                        "invalid corroborated canonical identity".into(),
                    ));
                }
            } else if !label(&target, "CanonicalFact")
                || string(&node, "canonical_node_id") != Some(target.id.as_str())
                || string(&node, "typed_fact_hash").is_none()
                || string(&node, "typed_fact_hash") != string(&target, "typed_fact_hash")
            {
                return Err(ArticleFoldError::InvalidAuthority(
                    "invalid fact alias witness".into(),
                ));
            }
            targets.insert(target.id.clone(), target);
        }
        match targets.len() {
            0 if label(&node, "AliasWitness") => {
                return Err(ArticleFoldError::InvalidAuthority(
                    "unbound alias witness".into(),
                ));
            }
            0 => return Ok(node),
            1 => node = targets.into_values().next().unwrap(),
            _ => return Err(ArticleFoldError::AmbiguousIdentity(node.id)),
        }
    }
    Err(ArticleFoldError::IncompleteRead(
        "canonical alias depth exceeded".into(),
    ))
}

fn resolve<R: ArticleRead + ?Sized>(
    read: &R,
    exact: &str,
    reads: &mut Reads,
) -> Result<ArticleNode, ArticleFoldError> {
    let direct = load(read, exact, reads)?;
    // A named record remains an explicit authority boundary even when an
    // admitted alias happens to use its identifier as a human-readable name.
    if matches!(direct, ArticleNode::Excluded) {
        return Ok(ArticleNode::Excluded);
    }
    let mut candidates = read.identity_candidates(exact)?;
    match direct {
        ArticleNode::Present(node) => candidates.nodes.push(node),
        ArticleNode::Excluded | ArticleNode::Missing => (),
    }
    let mut resolved = BTreeMap::new();
    for node in candidates.nodes {
        reads.add("node", &node.id)?;
        if !admitted(read, &node, reads)? {
            candidates.excluded = true;
            continue;
        }
        match canonicalize(read, node, reads) {
            Ok(node) => {
                resolved.insert(node.id.clone(), node);
            }
            Err(ArticleFoldError::Excluded) => candidates.excluded = true,
            Err(error) => return Err(error),
        }
    }
    // Name resolution is scoped: an unrelated private name cannot suppress an
    // admitted article or change its ambiguity. If no admitted candidate
    // remains, preserve exclusion so inaccessible targets never become red links.
    match resolved.len() {
        0 if candidates.excluded => Ok(ArticleNode::Excluded),
        0 => Ok(ArticleNode::Missing),
        1 => Ok(ArticleNode::Present(resolved.into_values().next().unwrap())),
        _ => Err(ArticleFoldError::AmbiguousIdentity(exact.into())),
    }
}

pub fn article_kind(node: &NodeRecord) -> String {
    if label(node, "CanonicalFact") {
        if let Some(kind) = node
            .properties
            .pointer("/fact/fact_type")
            .and_then(Value::as_str)
        {
            return kind.trim().to_ascii_lowercase();
        }
    }
    if let Some(kind) = string(node, "kind") {
        return kind.to_ascii_lowercase();
    }
    for (label_name, kind) in [
        ("MemoryProject", "project"),
        ("MemoryDocument", "document"),
        ("Person", "person"),
        ("Organization", "organization"),
        ("Project", "project"),
        ("Repo", "repo"),
        ("Repository", "repo"),
        ("Specification", "spec"),
        ("Spec", "spec"),
        ("Plan", "plan"),
        ("Goal", "goal"),
        ("Decision", "decision"),
        ("Place", "place"),
        ("Document", "document"),
        ("Tool", "tool"),
        ("Model", "model"),
    ] {
        if label(node, label_name) {
            return kind.into();
        }
    }
    "entity".into()
}

fn title(node: &NodeRecord) -> String {
    string(node, "title")
        .or_else(|| string(node, "name"))
        .or_else(|| string(node, "project_slug"))
        .or_else(|| {
            node.properties
                .pointer("/fact/canonical_key")
                .and_then(Value::as_str)
        })
        .unwrap_or(&node.id)
        .to_owned()
}

/// Lossless fallback identity. The HTTP boundary percent-encodes path segments
/// once; a title-to-slug heuristic would introduce unregistered collisions.
pub fn article_slug(node: &NodeRecord) -> String {
    string(node, "slug").unwrap_or(&node.id).to_owned()
}

fn link(node: &NodeRecord) -> Backlink {
    Backlink {
        entity_id: node.id.clone(),
        slug: article_slug(node),
        title: title(node),
    }
}

fn paragraph(text: &str) -> &str {
    let text = text.trim();
    text.find("\n\n")
        .or_else(|| text.find("\r\n\r\n"))
        .map(|i| &text[..i])
        .unwrap_or(text)
        .trim()
}

fn lead_text(node: &NodeRecord) -> String {
    string(node, "gist")
        .or_else(|| string(node, "l0_gist"))
        .map(str::to_owned)
        .unwrap_or_else(|| {
            paragraph(
                string(node, "overview")
                    .or_else(|| string(node, "summary"))
                    .or_else(|| string(node, "content"))
                    .or_else(|| {
                        node.properties
                            .pointer("/fact/statement")
                            .and_then(Value::as_str)
                    })
                    .unwrap_or(""),
            )
            .to_owned()
        })
}

fn sentence_text(text: &str, citations: &[ReferenceId]) -> Vec<Sentence> {
    // Split only at an explicit sentence terminator followed by whitespace,
    // preserving every byte of the source prose (no generated prose or LLM).
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if matches!(ch, '.' | '!' | '?')
            && chars.peek().is_none_or(|(_, next)| next.is_whitespace())
        {
            let end = index + ch.len_utf8();
            let value = text[start..end].trim();
            if !value.is_empty() {
                out.push(Sentence {
                    text: value.into(),
                    citations: citations.to_vec(),
                });
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(Sentence {
            text: tail.into(),
            citations: citations.to_vec(),
        });
    }
    out
}

fn citations<R: ArticleRead + ?Sized>(
    read: &R,
    node: &NodeRecord,
    reads: &mut Reads,
    references: &mut BTreeMap<ReferenceId, Reference>,
) -> Result<Vec<ReferenceId>, ArticleFoldError> {
    let mut ids = BTreeSet::new();
    for evidence in read.evidence(node)? {
        let mut visible = true;
        for id in &evidence.dependency_node_ids {
            match load(read, id, reads)? {
                ArticleNode::Present(_) => (),
                ArticleNode::Excluded => visible = false,
                ArticleNode::Missing => {
                    return Err(ArticleFoldError::IncompleteRead(
                        "receipt dependency is absent".into(),
                    ));
                }
            }
        }
        if !visible {
            continue;
        }
        let mut reference = evidence.reference;
        // Receipt identity, not an alias chosen by an adapter, is the footnote
        // deduplication authority.
        reference.id = ReferenceId(reference.receipt_id.clone());
        if reference.id.0.is_empty()
            || reference.receipt_id.is_empty()
            || reference.source_kind.is_empty()
            || reference.replay_handle.is_empty()
        {
            return Err(ArticleFoldError::InvalidAuthority(
                "incomplete evidence receipt".into(),
            ));
        }
        if references
            .get(&reference.id)
            .is_some_and(|existing| existing != &reference)
        {
            return Err(ArticleFoldError::InvalidAuthority(
                "receipt identity collision".into(),
            ));
        }
        ids.insert(reference.id.clone());
        references.insert(reference.id.clone(), reference);
    }
    Ok(ids.into_iter().collect())
}

fn section_id(node: &NodeRecord) -> &'static str {
    match article_kind(node).as_str() {
        "decision" => "decisions",
        "spec" | "specification" => "specifications",
        "plan" | "goal" => "plans",
        "discussion" | "room" | "thread" => "discussions",
        "evidence" | "derivation" | "receipt" => "evidence",
        _ if node.labels.iter().any(|s| {
            matches!(
                s.as_str(),
                "CoordinationMessage" | "CoordinationRoom" | "ChatThread"
            )
        }) =>
        {
            "discussions"
        }
        _ if label(node, "Derivation") => "evidence",
        _ => "related",
    }
}

fn infobox_rows(kind: &str) -> &'static [(&'static str, &'static str)] {
    match kind {
        "repo" => &[
            ("owner", "Owner"),
            ("default_branch", "Default branch"),
            ("primary_language", "Primary language"),
            ("active_plans", "Active plans"),
            ("last_commit", "Last commit"),
        ],
        "person" => &[
            ("role", "Role"),
            ("organization", "Organization"),
            ("projects", "Projects"),
        ],
        "project" => &[
            ("repos", "Repositories"),
            ("owner", "Owner"),
            ("status", "Status"),
            ("active_plans", "Active plans"),
        ],
        "organization" => &[
            ("owner", "Owner"),
            ("place", "Place"),
            ("projects", "Projects"),
        ],
        "plan" | "goal" => &[
            ("owner", "Owner"),
            ("status", "Status"),
            ("project", "Project"),
        ],
        "spec" | "decision" => &[
            ("status", "Status"),
            ("project", "Project"),
            ("author", "Author"),
        ],
        "document" => &[
            ("author", "Author"),
            ("status", "Status"),
            ("project", "Project"),
        ],
        "tool" | "model" => &[
            ("owner", "Owner"),
            ("version", "Version"),
            ("status", "Status"),
        ],
        "place" => &[("country", "Country"), ("region", "Region")],
        _ => &[],
    }
}

fn typed_property<'a>(node: &'a NodeRecord, key: &str) -> Option<&'a Value> {
    node.properties.get(key).or_else(|| match key {
        "primary_language" => node.properties.get("language"),
        "author" => node.properties.get("actor_id"),
        "project" => node.properties.get("project_slug"),
        _ => None,
    })
}

fn info_relation(
    key: &str,
    edge: &EdgeRecord,
    neighbor: &NodeRecord,
    roots: &BTreeMap<String, NodeRecord>,
) -> bool {
    let relation = edge.edge_type.to_ascii_lowercase();
    match key {
        "owner" => {
            roots.contains_key(&edge.from_id) && matches!(relation.as_str(), "owned_by" | "owner")
        }
        "organization" => {
            roots.contains_key(&edge.from_id)
                && relation == "member_of"
                && article_kind(neighbor) == "organization"
        }
        "projects" | "project" => {
            roots.contains_key(&edge.from_id)
                && matches!(
                    relation.as_str(),
                    "works_on" | "in_project" | "memory_in_project"
                )
                && article_kind(neighbor) == "project"
        }
        "active_plans" => {
            article_kind(neighbor) == "plan"
                && !matches!(
                    string(neighbor, "status"),
                    Some("completed" | "done" | "archived" | "cancelled")
                )
        }
        "repos" => article_kind(neighbor) == "repo",
        _ => false,
    }
}

/// Fold a navigated identity. Missing and excluded targets remain distinct:
/// only Missing can become a red link at the effect boundary.
/// A transient eligibility result. This is not a published article, history
/// revision, or commit; preparation must never invent those authorities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArticleEligibility {
    pub canonical_entity_id: String,
    pub eligible: bool,
    pub dependency_ids: BTreeSet<String>,
    pub dependency_keys: BTreeSet<String>,
}

struct ArticleCluster {
    canonical: NodeRecord,
    roots: BTreeMap<String, NodeRecord>,
    display: NodeRecord,
    kind: String,
    cluster: BTreeMap<String, NodeRecord>,
    neighbors: BTreeMap<String, NodeRecord>,
    backlinks: BTreeMap<String, Backlink>,
    see_also: BTreeMap<String, Backlink>,
    red_links: BTreeSet<RedLink>,
    relationships: Vec<(EdgeRecord, NodeRecord)>,
    stub: bool,
}

/// Uses the exact canonical witness, admission, and degree/source rules used
/// by fold_article, without asking a transient prepared view for a revision.
pub fn article_eligibility<R: ArticleRead + ?Sized>(
    read: &R,
    entity_id: &str,
) -> Result<ArticleEligibility, ArticleFoldError> {
    let mut reads = Reads::default();
    let cluster = fold_cluster(read, entity_id, &mut reads)?;
    Ok(ArticleEligibility {
        canonical_entity_id: cluster.canonical.id,
        eligible: !cluster.stub,
        dependency_ids: reads.ids,
        dependency_keys: reads.keys,
    })
}

fn fold_cluster<R: ArticleRead + ?Sized>(
    read: &R,
    entity_id: &str,
    reads: &mut Reads,
) -> Result<ArticleCluster, ArticleFoldError> {
    let canonical = match resolve(read, entity_id, reads)? {
        ArticleNode::Present(node) => node,
        ArticleNode::Missing => return Err(ArticleFoldError::Missing(entity_id.into())),
        ArticleNode::Excluded => return Err(ArticleFoldError::Excluded),
    };
    let mut roots = BTreeMap::from([(canonical.id.clone(), canonical.clone())]);
    // Existing CanonicalEntity is an identity record without title/gist. Read
    // corroborated members, preserving deterministic display precedence.
    if label(&canonical, "CanonicalEntity") {
        for id in strings(canonical.properties.get("canonical_for")) {
            match load(read, &id, reads)? {
                ArticleNode::Present(member) => {
                    if canonicalize(read, member.clone(), reads)?.id != canonical.id {
                        return Err(ArticleFoldError::InvalidAuthority(
                            "canonical membership lacks witness".into(),
                        ));
                    }
                    roots.insert(id, member);
                }
                ArticleNode::Excluded => return Err(ArticleFoldError::Excluded),
                ArticleNode::Missing => {
                    return Err(ArticleFoldError::InvalidAuthority(
                        "canonical member missing".into(),
                    ));
                }
            }
        }
    }
    let display = if string(&canonical, "title").is_some() || !label(&canonical, "CanonicalEntity")
    {
        &canonical
    } else {
        roots
            .values()
            .find(|n| n.id != canonical.id)
            .unwrap_or(&canonical)
    };
    let display = display.clone();
    let kind = article_kind(&display);
    let mut cluster = roots.clone();
    let mut neighbors = BTreeMap::new();
    let mut inbound = BTreeSet::new();
    let mut provenance = BTreeSet::new();
    let mut backlinks = BTreeMap::new();
    let mut see_also = BTreeMap::new();
    let mut red_links = BTreeSet::new();
    let mut relationships = Vec::new();
    for root in roots.values() {
        for direction in [Direction::Out, Direction::In] {
            for edge in read.edges(&root.id, direction.clone())? {
                if edge.tombstone {
                    continue;
                }
                reads.add("edge", &edge.id)?;
                let other_id = match direction {
                    Direction::Out => &edge.to_id,
                    Direction::In => &edge.from_id,
                };
                let other = match resolve(read, other_id, reads)? {
                    ArticleNode::Present(node) => node,
                    ArticleNode::Missing if direction == Direction::Out && edge_allowed(&edge) => {
                        red_links.insert(RedLink {
                            target: other_id.clone(),
                            title: other_id.clone(),
                            referring_entity_id: canonical.id.clone(),
                        });
                        continue;
                    }
                    ArticleNode::Missing | ArticleNode::Excluded => continue,
                };
                if roots.contains_key(&other.id) {
                    continue;
                }
                if direction == Direction::In {
                    // Degree counts distinct persisted incoming edges.
                    inbound.insert(edge.id.clone());
                    if let Some(source) = edge
                        .provenance
                        .as_ref()
                        .and_then(|p| p.source_id.as_deref())
                        .or_else(|| {
                            edge.properties
                                .get("source_cluster")
                                .and_then(Value::as_str)
                        })
                        .or_else(|| edge.properties.get("source").and_then(Value::as_str))
                    {
                        if !source.trim().is_empty() {
                            provenance.insert(source.to_owned());
                        }
                    }
                    backlinks.insert(other.id.clone(), link(&other));
                }
                if edge_allowed(&edge) {
                    relationships.push((edge, other.clone()));
                    see_also.insert(other.id.clone(), link(&other));
                    cluster.insert(other.id.clone(), other.clone());
                    neighbors.insert(other.id.clone(), other);
                }
            }
        }
        // Runtime writes the links field even when the missing target prevented
        // MEMORY_RELATES materialization. These are real red links, not guesses.
        for target in strings(root.properties.get("links")) {
            match resolve(read, &target, reads)? {
                ArticleNode::Present(node) if !roots.contains_key(&node.id) => {
                    see_also.insert(node.id.clone(), link(&node));
                    cluster.insert(node.id.clone(), node.clone());
                    neighbors.insert(node.id.clone(), node);
                }
                ArticleNode::Missing => {
                    red_links.insert(RedLink {
                        title: target.clone(),
                        target,
                        referring_entity_id: canonical.id.clone(),
                    });
                }
                _ => (),
            }
        }
    }
    let stub =
        !ARTICLE_KINDS.contains(&kind.as_str()) && !(inbound.len() >= 5 && provenance.len() >= 2);
    Ok(ArticleCluster {
        canonical,
        roots,
        display,
        kind,
        cluster,
        neighbors,
        backlinks,
        see_also,
        red_links,
        relationships,
        stub,
    })
}

pub fn fold_article<R: ArticleRead + ?Sized>(
    read: &R,
    entity_id: &str,
    tier: ArticleTier,
) -> Result<ArticleProjection, ArticleFoldError> {
    let mut reads = Reads::default();
    let ArticleCluster {
        canonical,
        roots,
        display,
        kind,
        cluster,
        mut neighbors,
        backlinks,
        see_also,
        red_links,
        relationships,
        stub,
    } = fold_cluster(read, entity_id, &mut reads)?;
    let display = &display;
    let mut references = BTreeMap::new();
    let lead_citations = citations(read, display, &mut reads, &mut references)?;
    let lead = sentence_text(&lead_text(display), &lead_citations);
    let mut infobox = Vec::new();
    if !stub && tier != ArticleTier::Gist {
        for &(key, row_label) in infobox_rows(&kind) {
            let mut values = Vec::new();
            if let Some(value) = typed_property(display, key) {
                let texts = match value {
                    Value::String(s) => vec![s.clone()],
                    Value::Array(_) => strings(Some(value)),
                    Value::Number(n) => vec![n.to_string()],
                    Value::Bool(b) => vec![b.to_string()],
                    _ => vec![],
                };
                for text in texts.into_iter().filter(|s| !s.trim().is_empty()) {
                    values.push(InfoboxValue::Text { text });
                }
            }
            let mut links = BTreeMap::new();
            for (edge, neighbor) in &relationships {
                if info_relation(key, edge, neighbor, &roots) {
                    links.insert(neighbor.id.clone(), link(neighbor));
                }
            }
            values.extend(links.into_values().map(|link| InfoboxValue::Link { link }));
            if !values.is_empty() {
                infobox.push(InfoboxRow {
                    id: key.into(),
                    label: row_label.into(),
                    values,
                    citations: lead_citations.clone(),
                });
            }
        }
    }
    let mut pins = BTreeMap::new();
    for pin in read.pinned_sections(&canonical.id)? {
        if pin.block.document_id.trim().is_empty()
            || pin.block.block_id.trim().is_empty()
            || pin.block.content_digest.trim().is_empty()
            || pin.dependency_node_ids.is_empty()
        {
            return Err(ArticleFoldError::InvalidAuthority(
                "section pin lacks document authority".into(),
            ));
        }
        if !ARTICLE_SECTION_ORDER.contains(&pin.section_id.as_str())
            || pins.contains_key(&pin.section_id)
        {
            return Err(ArticleFoldError::InvalidAuthority(
                "unknown or duplicated section pin".into(),
            ));
        }
        for dependency in &pin.dependency_node_ids {
            match load(read, dependency, &mut reads)? {
                ArticleNode::Present(_) => (),
                ArticleNode::Excluded => return Err(ArticleFoldError::Excluded),
                ArticleNode::Missing => {
                    return Err(ArticleFoldError::IncompleteRead(
                        "pinned document dependency missing".into(),
                    ));
                }
            }
        }
        pins.insert(pin.section_id.clone(), pin);
    }
    // Canonical memory articles must expose the stored document body, not
    // merely its gist. The fixed section uses the same source projection as
    // related documents; this does not create an authored Wiki document.
    if string(display, "content").is_some() {
        neighbors.insert(display.id.clone(), display.clone());
    }
    let mut sections = Vec::new();
    if !stub && tier != ArticleTier::Gist {
        for (section, heading) in ARTICLE_SECTION_ORDER.into_iter().zip([
            "Decisions",
            "Specifications",
            "Plans",
            "Discussions",
            "Evidence",
            "Related",
        ]) {
            if let Some(pin) = pins.remove(section) {
                sections.push(Section {
                    id: section.into(),
                    level: 2,
                    title: heading.into(),
                    body: SectionBody::Empty,
                    pinned: Some(pin.block),
                    citations: vec![],
                });
                continue;
            }
            let mut entries = Vec::new();
            let mut section_citations = BTreeSet::new();
            for neighbor in neighbors
                .values()
                .filter(|node| section_id(node) == section)
            {
                let ids = citations(read, neighbor, &mut reads, &mut references)?;
                let text = if tier == ArticleTier::Full {
                    string(neighbor, "content")
                        .map(str::to_owned)
                        .unwrap_or_else(|| lead_text(neighbor))
                } else {
                    lead_text(neighbor)
                };
                let sentences = sentence_text(&text, &ids);
                if sentences.is_empty() {
                    continue;
                }
                section_citations.extend(ids);
                entries.push(SectionEntry {
                    source_text: text,
                    entity_id: neighbor.id.clone(),
                    title: title(neighbor),
                    slug: article_slug(neighbor),
                    sentences,
                });
            }
            if !entries.is_empty() {
                sections.push(Section {
                    id: section.into(),
                    level: 2,
                    title: heading.into(),
                    body: SectionBody::Entries { entries },
                    pinned: None,
                    citations: section_citations.into_iter().collect(),
                });
            }
        }
    }
    let categories = if stub {
        vec![]
    } else {
        let mut labels = BTreeSet::from([kind.clone()]);
        for node in roots.values() {
            labels.extend(node.labels.iter().cloned());
        }
        labels
            .into_iter()
            .map(|label| Category {
                id: label.clone(),
                label,
            })
            .collect()
    };
    // Keep only receipts actually cited by emitted prose or rows. Pinned and
    // gist-only sections cannot leave unrelated footnotes in the projection.
    let used = lead
        .iter()
        .flat_map(|s| &s.citations)
        .chain(infobox.iter().flat_map(|r| &r.citations))
        .chain(sections.iter().flat_map(|s| &s.citations))
        .cloned()
        .collect::<BTreeSet<_>>();
    references.retain(|id, _| used.contains(id));
    let revision = read.revision(&reads.keys)?;
    Ok(ArticleProjection {
        schema: ARTICLE_SCHEMA.into(),
        read_scope: Default::default(),
        links: None,
        entity_id: canonical.id.clone(),
        title: title(display),
        slug: article_slug(&canonical),
        kind,
        lead,
        hatnote: (entity_id != canonical.id && entity_id != article_slug(&canonical)).then(|| {
            Hatnote {
                alias: entity_id.into(),
                canonical_entity_id: canonical.id.clone(),
                canonical_title: title(display),
            }
        }),
        infobox,
        sections,
        references: references.into_values().collect(),
        categories,
        red_links: red_links.into_iter().collect(),
        backlinks: backlinks.into_values().take(ARTICLE_LINK_LIMIT).collect(),
        see_also: see_also.into_values().take(ARTICLE_LINK_LIMIT).collect(),
        revision,
        stub,
        cluster_ids: cluster.into_keys().collect(),
        dependency_ids: reads.ids.into_iter().collect(),
        dependency_keys: reads.keys.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests;
