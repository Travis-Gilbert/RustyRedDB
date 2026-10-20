//! Neutral link evidence and scalar feature contracts, shared below core,
//! the signal pipeline, and the five candidate generators.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    Inline,
    SeeAlso,
    CrossScope,
    Typed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JustificationKind {
    #[default]
    Identity,
    Mention,
    Similarity,
    Structure,
    Inference,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "evidence", rename_all = "snake_case")]
pub enum Justification {
    Identity(IdentityEvidence),
    Mention(MentionEvidence),
    Similarity(SimilarityEvidence),
    Structure(StructureEvidence),
    Inference(InferenceEvidence),
}
impl Justification {
    pub fn discriminant(&self) -> JustificationKind {
        match self {
            Self::Identity(_) => JustificationKind::Identity,
            Self::Mention(_) => JustificationKind::Mention,
            Self::Similarity(_) => JustificationKind::Similarity,
            Self::Structure(_) => JustificationKind::Structure,
            Self::Inference(_) => JustificationKind::Inference,
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        let text = |s: &str| !s.trim().is_empty();
        let strings = |s: &[String]| {
            !s.is_empty()
                && s.iter().all(|s| text(s))
                && s.iter().collect::<std::collections::BTreeSet<_>>().len() == s.len()
        };
        let valid = match self {
            Self::Identity(IdentityEvidence::Identifier {
                field,
                value,
                edge_def_id,
                ..
            }) => text(field) && text(value) && text(edge_def_id),
            Self::Identity(IdentityEvidence::Cluster {
                entity_id,
                match_edge_receipt_ids,
                saturation_receipt_id,
            }) => text(entity_id) && strings(match_edge_receipt_ids) && text(saturation_receipt_id),
            Self::Mention(e) => {
                text(&e.span.block_id)
                    && e.span.length > 0
                    && e.span.offset.checked_add(e.span.length).is_some()
                    && e.span.length == e.anchor_text.len() as u64
                    && text(&e.anchor_text)
                    && text(&e.alias_id)
                    && probability(e.keyphraseness)
                    && probability(e.commonness)
                    && e.occurrences > 0
                    && (e.anchor_links > 0 || e.commonness == 0.0)
                    && e.linked_occurrences <= e.occurrences
                    && e.target_links <= e.anchor_links
                    && e.keyphraseness == e.linked_occurrences as f64 / e.occurrences as f64
                    && e.commonness
                        == if e.anchor_links == 0 {
                            0.0
                        } else {
                            e.target_links as f64 / e.anchor_links as f64
                        }
            }
            Self::Similarity(SimilarityEvidence::Embedding {
                embedding_distance,
                embedding_model,
                index_receipt_id,
            }) => {
                embedding_distance.is_finite()
                    && *embedding_distance >= 0.0
                    && text(embedding_model)
                    && text(index_receipt_id)
            }
            Self::Similarity(SimilarityEvidence::CoCitation {
                shared_receipt_ids,
                cocitation_count,
            }) => {
                strings(shared_receipt_ids)
                    && *cocitation_count as usize == shared_receipt_ids.len()
            }
            Self::Structure(e) => {
                strings(&e.common_neighbor_ids)
                    && strings(&e.edge_kinds)
                    && e.adamic_adar.is_finite()
                    && e.adamic_adar >= 0.0
                    && text(&e.offload_receipt_id)
            }
            Self::Inference(e) => {
                strings(&e.intermediate_ids)
                    && strings(&e.path_kinds)
                    && e.shared_intermediates as usize == e.intermediate_ids.len()
                    && text(&e.offload_receipt_id)
                    && text(&e.derivation_rule_pack)
                    && text(&e.derivation_fact_pack)
                    && !e.absences.is_empty()
                    && e.absences.iter().all(|a| a.validate().is_ok())
            }
        };
        if valid {
            Ok(())
        } else {
            Err("link justification lacks valid typed evidence".into())
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum IdentityEvidence {
    Identifier {
        field: String,
        value: String,
        edge_def_id: String,
        definition_version: u32,
    },
    Cluster {
        entity_id: String,
        match_edge_receipt_ids: Vec<String>,
        saturation_receipt_id: String,
    },
}
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LinkSpan {
    pub block_id: String,
    pub offset: u64,
    pub length: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MentionEvidence {
    pub span: LinkSpan,
    pub anchor_text: String,
    pub keyphraseness: f64,
    pub commonness: f64,
    pub alias_id: String,
    pub occurrences: u64,
    pub linked_occurrences: u64,
    pub anchor_links: u64,
    pub target_links: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum SimilarityEvidence {
    Embedding {
        embedding_distance: f64,
        embedding_model: String,
        index_receipt_id: String,
    },
    CoCitation {
        shared_receipt_ids: Vec<String>,
        cocitation_count: u32,
    },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructureEvidence {
    pub common_neighbor_ids: Vec<String>,
    pub adamic_adar: f64,
    pub edge_kinds: Vec<String>,
    pub offload_receipt_id: String,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LinkScalar {
    String(String),
    I64(i64),
    F64(f64),
    Bool(bool),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AbsentRelationEvidence {
    pub relation: String,
    pub pattern: Vec<Option<LinkScalar>>,
    pub closed_relation: String,
    pub input_facts: String,
    pub fact_pack: String,
    pub rule_pack: String,
    pub stratum: usize,
}
impl AbsentRelationEvidence {
    pub fn validate(&self) -> Result<(), String> {
        if [
            &self.relation,
            &self.closed_relation,
            &self.input_facts,
            &self.fact_pack,
            &self.rule_pack,
        ]
        .iter()
        .any(|s| s.trim().is_empty())
            || self.stratum == 0
            || self
                .pattern
                .iter()
                .flatten()
                .any(|a| matches!(a,LinkScalar::F64(f) if !f.is_finite()))
        {
            return Err("invalid closed-input absence evidence".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InferenceEvidence {
    pub intermediate_ids: Vec<String>,
    pub path_kinds: Vec<String>,
    pub shared_intermediates: u32,
    pub offload_receipt_id: String,
    pub derivation_rule_pack: String,
    pub derivation_fact_pack: String,
    pub absences: Vec<AbsentRelationEvidence>,
}

/// The scorer receives only these scalar values and one closed discriminator.
/// No anchor text, node ID, receipt payload or other evidence enters this type.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LinkFeatures {
    pub identifier_match: bool,
    pub keyphraseness: f64,
    pub commonness: f64,
    pub lexical_overlap: f64,
    pub entity_edge_overlap: f64,
    pub embedding_distance: Option<f64>,
    pub cocitation_count: u32,
    pub common_neighbors: u32,
    pub adamic_adar: f64,
    pub shared_intermediates: u32,
    pub path_length: u8,
    pub graph_degree_src: usize,
    pub graph_degree_dst: usize,
    pub cross_scope: bool,
    pub justification: JustificationKind,
}
impl LinkFeatures {
    pub fn validate(&self) -> Result<(), String> {
        if ![
            self.keyphraseness,
            self.commonness,
            self.lexical_overlap,
            self.entity_edge_overlap,
        ]
        .into_iter()
        .all(probability)
            || !self.adamic_adar.is_finite()
            || self.adamic_adar < 0.0
            || self
                .embedding_distance
                .is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            return Err("invalid link feature scalar".into());
        }
        Ok(())
    }
}
fn probability(v: f64) -> bool {
    v.is_finite() && (0.0..=1.0).contains(&v)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LinkReadStamp {
    pub scope: String,
    pub graph_root: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LinkCandidate {
    pub candidate_id: String,
    pub src: String,
    pub dst: String,
    pub kind: LinkKind,
    pub generator: String,
    pub justification: Justification,
    pub features: LinkFeatures,
    pub read_stamp: LinkReadStamp,
}
impl<'de> Deserialize<'de> for LinkCandidate {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            candidate_id: String,
            src: String,
            dst: String,
            kind: LinkKind,
            generator: String,
            justification: Justification,
            features: LinkFeatures,
            read_stamp: LinkReadStamp,
        }
        let wire = Wire::deserialize(deserializer)?;
        let candidate = Self {
            candidate_id: wire.candidate_id,
            src: wire.src,
            dst: wire.dst,
            kind: wire.kind,
            generator: wire.generator,
            justification: wire.justification,
            features: wire.features,
            read_stamp: wire.read_stamp,
        };
        candidate.validate().map_err(serde::de::Error::custom)?;
        Ok(candidate)
    }
}
impl LinkCandidate {
    pub fn validate(&self) -> Result<(), String> {
        if [
            &self.candidate_id,
            &self.src,
            &self.dst,
            &self.generator,
            &self.read_stamp.scope,
            &self.read_stamp.graph_root,
        ]
        .iter()
        .any(|s| s.trim().is_empty())
            || self.src == self.dst
        {
            return Err("link requires distinct bound endpoints and a generator".into());
        }
        self.justification.validate()?;
        self.features.validate()?;
        if self.features.justification != self.justification.discriminant() {
            return Err("link evidence and feature discriminants disagree".into());
        }
        Ok(())
    }
}

pub const LINKED_TO_TYPE: &str = "LINKED_TO";
pub const LINKED_TO_SCHEMA: &str = "theorem-linked-to/1";
pub const LINKED_TO_PROPERTIES: &[&str] = &[
    "schema",
    "kind",
    "justification",
    "generator",
    "score",
    "model_version",
    "receipt_id",
    "evidence",
    "visibility",
    "written_at_graph_version",
];

/// The policy payload remains owned by DATAWAVE. Core can validate and stamp
/// the shared record header without depending upward on an intake or UI crate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkedToRecord<V> {
    pub schema: String,
    pub kind: LinkKind,
    pub justification: JustificationKind,
    pub generator: String,
    pub score: f64,
    pub model_version: String,
    pub receipt_id: String,
    pub evidence: Justification,
    pub visibility: V,
    /// None exists only in an uncommitted mutation. The graph writer stamps
    /// its actual allocated edge version before hashing or publishing it.
    pub written_at_graph_version: Option<u64>,
}
impl<V> LinkedToRecord<V> {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != LINKED_TO_SCHEMA
            || !probability(self.score)
            || [&self.generator, &self.model_version, &self.receipt_id]
                .iter()
                .any(|value| value.trim().is_empty())
            || self.justification != self.evidence.discriminant()
            || self.written_at_graph_version == Some(0)
        {
            return Err("invalid canonical LINKED_TO record".into());
        }
        self.evidence.validate()
    }
}

/// One directed link per endpoint tuple and kind, independent of generator,
/// model, score, or the receipt that last updated it.
pub fn linked_to_identity_bytes(src: &str, dst: &str, kind: LinkKind) -> Vec<u8> {
    serde_json::to_vec(&(LINKED_TO_SCHEMA, src, dst, kind))
        .expect("link identity string tuple serializes")
}
