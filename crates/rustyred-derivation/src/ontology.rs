//! Edge ontology: the signature every valuation interprets.
//!
//! The edge ontology and the valuation algebra are related as signature and
//! interpretations. This module owns the signature. Each semiring in
//! [`crate::semiring`] remains an interpretation of the *whole* signature —
//! there is no per-edge-type semiring and no per-semiring edge type.
//!
//! Every edge type declares exactly one [`AlgebraicClass`]. New edge types are
//! added by declaring a class and a registry row; an edge type without a class
//! is rejected at construction rather than defaulted, and an unknown edge type
//! resolves to `None` rather than to a default class. Default trust is never
//! granted silently.
//!
//! The registry lives here rather than in `rustyred-thg-core` because the
//! dependency direction is fixed by the workspace: core depends on derivation,
//! derivation does not see `EdgeRecord`. The registry therefore speaks in
//! edge-type strings, epistemic-type strings, and `f64` annotations. Core
//! provides the thin `EdgeRecord` adapter.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

/// How an edge type participates in valuation.
///
/// The four classes match what the code already did implicitly before the
/// registry existed:
///
/// - [`AlgebraicClass::MonotonePositive`] — semiring generators. Composition is
///   monotone, which is exactly what a semiring fold can express.
/// - [`AlgebraicClass::Defeating`] — dual generators. Semiring addition and
///   multiplication are monotone and there is no subtraction, so defeat is not
///   expressible in a semiring at all; it mints a negative indeterminate in
///   [`crate::dual`] instead.
/// - [`AlgebraicClass::Equivalence`] — quotient structure, never valued.
/// - [`AlgebraicClass::Structural`] — plumbing, never valued.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgebraicClass {
    MonotonePositive,
    Defeating,
    Equivalence,
    Structural,
}

/// Whether an edge keeps its stored orientation or may be traversed from
/// either endpoint without changing its meaning.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeDirection {
    #[default]
    Directed,
    Symmetric,
}

impl EdgeDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Directed => "directed",
            Self::Symmetric => "symmetric",
        }
    }
}

impl AlgebraicClass {
    /// Whether this class mints a generator in the free dual layer.
    ///
    /// `Equivalence` and `Structural` edges mint nothing and pass through a
    /// fold as the multiplicative identity.
    pub fn is_valued(&self) -> bool {
        matches!(self, Self::MonotonePositive | Self::Defeating)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MonotonePositive => "monotone_positive",
            Self::Defeating => "defeating",
            Self::Equivalence => "equivalence",
            Self::Structural => "structural",
        }
    }
}

/// One row of the edge ontology.
///
/// `class` carries no serde default on purpose: a declaration that omits the
/// algebraic class fails to deserialize instead of silently acquiring one.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EdgeTypeSpec {
    pub canonical: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub class: AlgebraicClass,
    /// The `record.rs` `EpistemicType` token this canonical name corresponds
    /// to, when one exists (`"supports"`, `"undercuts"`, `"contradicts"`,
    /// `"tension"`, `"derives"`, `"cites"`). `None` for `Equivalence` and
    /// `Structural` rows.
    #[serde(default)]
    pub epistemic_type: Option<String>,
    /// Traversal semantics independent of the stored `from`/`to` shape.
    #[serde(default)]
    pub direction: EdgeDirection,
    /// Edge property whose value participates in evidence weighting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_property: Option<String>,
}

impl EdgeTypeSpec {
    pub fn new(
        canonical: impl Into<String>,
        class: AlgebraicClass,
        aliases: impl IntoIterator<Item = impl Into<String>>,
        epistemic_type: Option<&str>,
    ) -> Self {
        Self {
            canonical: canonical.into(),
            aliases: aliases.into_iter().map(Into::into).collect(),
            class,
            epistemic_type: epistemic_type.map(str::to_string),
            direction: EdgeDirection::Directed,
            weight_property: None,
        }
    }

    #[must_use]
    pub fn with_direction(mut self, direction: EdgeDirection) -> Self {
        self.direction = direction;
        self
    }

    #[must_use]
    pub fn with_weight_property(mut self, property: impl Into<String>) -> Self {
        self.weight_property = Some(property.into());
        self
    }
}

/// Construction refusals. Each names the colliding tokens.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum OntologyError {
    #[error("edge type declares an empty canonical name")]
    EmptyCanonical,
    #[error("duplicate canonical edge type: {tokens:?}")]
    DuplicateCanonical { tokens: Vec<String> },
    #[error("alias collides with a distinct edge type: {tokens:?}")]
    AliasCollision { tokens: Vec<String> },
}

/// Interned, case-insensitive index from edge-type token to [`EdgeTypeSpec`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EdgeOntology {
    specs: Vec<EdgeTypeSpec>,
    /// Case-folded token -> index into `specs`. Canonical names and aliases
    /// both land here.
    index: BTreeMap<String, usize>,
}

fn fold_token(token: &str) -> String {
    token.trim().to_ascii_lowercase()
}

impl EdgeOntology {
    /// Build a registry, rejecting duplicate canonical names and alias
    /// collisions across distinct rows.
    ///
    /// An alias that case-folds onto its own row's canonical name is a no-op,
    /// not a collision: the default registry deliberately lists case variants
    /// (`supports`, `SUPPORTS`) so that every string the epistemic pass has
    /// ever matched appears verbatim in the table.
    pub fn from_specs(specs: Vec<EdgeTypeSpec>) -> Result<Self, OntologyError> {
        let mut index: BTreeMap<String, usize> = BTreeMap::new();

        for (position, spec) in specs.iter().enumerate() {
            let key = fold_token(&spec.canonical);
            if key.is_empty() {
                return Err(OntologyError::EmptyCanonical);
            }
            if let Some(&prior) = index.get(&key) {
                return Err(OntologyError::DuplicateCanonical {
                    tokens: vec![specs[prior].canonical.clone(), spec.canonical.clone()],
                });
            }
            index.insert(key, position);
        }

        for (position, spec) in specs.iter().enumerate() {
            for alias in &spec.aliases {
                let key = fold_token(alias);
                if key.is_empty() {
                    return Err(OntologyError::AliasCollision {
                        tokens: vec![spec.canonical.clone(), alias.clone()],
                    });
                }
                match index.get(&key) {
                    // Idempotent: a case variant of this row's own canonical
                    // name, or the same alias listed twice.
                    Some(&owner) if owner == position => {}
                    Some(&owner) => {
                        return Err(OntologyError::AliasCollision {
                            tokens: vec![
                                alias.clone(),
                                spec.canonical.clone(),
                                specs[owner].canonical.clone(),
                            ],
                        })
                    }
                    None => {
                        index.insert(key, position);
                    }
                }
            }
        }

        Ok(Self { specs, index })
    }

    /// Case-insensitive; canonical names and aliases both resolve.
    /// Unknown edge types return `None`, never a default class.
    pub fn resolve(&self, edge_type: &str) -> Option<&EdgeTypeSpec> {
        self.index
            .get(&fold_token(edge_type))
            .map(|&position| &self.specs[position])
    }

    pub fn class_of(&self, edge_type: &str) -> Option<AlgebraicClass> {
        self.resolve(edge_type).map(|spec| spec.class)
    }

    /// Rows in declaration order.
    pub fn specs(&self) -> impl Iterator<Item = &EdgeTypeSpec> {
        self.specs.iter()
    }

    /// Canonical names of every row in `class`, in declaration order.
    ///
    /// This is what `StructuralEpistemicConfig` uses to derive its default
    /// support and attack edge-type lists instead of hardcoding vectors.
    pub fn canonical_names_in_class(&self, class: AlgebraicClass) -> Vec<String> {
        self.specs
            .iter()
            .filter(|spec| spec.class == class)
            .map(|spec| spec.canonical.clone())
            .collect()
    }

    /// Every token spelling that resolves to a row in `class`, canonical names
    /// and aliases alike, deduplicated only by exact string.
    ///
    /// Distinct from [`EdgeOntology::tokens_in_class`], which folds case
    /// variants together because resolution is case-insensitive. A graph
    /// neighbor query matches an edge type exactly, so a traversal that needs
    /// to reach edges stored as `CONTRADICTS` and as `contradicts` has to ask
    /// for both spellings.
    pub fn edge_type_spellings_in_class(&self, class: AlgebraicClass) -> Vec<String> {
        let mut seen: BTreeMap<String, ()> = BTreeMap::new();
        let mut tokens = Vec::new();
        for spec in self.specs.iter().filter(|spec| spec.class == class) {
            for token in std::iter::once(&spec.canonical).chain(spec.aliases.iter()) {
                let trimmed = token.trim();
                if !trimmed.is_empty() && seen.insert(trimmed.to_string(), ()).is_none() {
                    tokens.push(trimmed.to_string());
                }
            }
        }
        tokens
    }

    /// Every token that resolves to a row in `class` — canonical names and
    /// aliases, in declaration order, deduplicated by case folding.
    pub fn tokens_in_class(&self, class: AlgebraicClass) -> Vec<String> {
        let mut seen: BTreeMap<String, ()> = BTreeMap::new();
        let mut tokens = Vec::new();
        for spec in self.specs.iter().filter(|spec| spec.class == class) {
            for token in std::iter::once(&spec.canonical).chain(spec.aliases.iter()) {
                if seen.insert(fold_token(token), ()).is_none() {
                    tokens.push(token.clone());
                }
            }
        }
        tokens
    }
}

/// The default registry rows.
///
/// Every string matched anywhere in `rustyred-thg-core`'s `epistemic.rs`
/// appears here as a canonical name or an alias.
pub fn default_specs() -> Vec<EdgeTypeSpec> {
    vec![
        EdgeTypeSpec::new(
            "Supports",
            AlgebraicClass::MonotonePositive,
            ["supports", "SUPPORTS"],
            Some("supports"),
        ),
        EdgeTypeSpec::new(
            "Cites",
            AlgebraicClass::MonotonePositive,
            ["CITES", "cites"],
            Some("cites"),
        ),
        EdgeTypeSpec::new(
            "Derives",
            AlgebraicClass::MonotonePositive,
            ["DERIVED_FROM", "derived_from", "derives"],
            Some("derives"),
        ),
        EdgeTypeSpec::new(
            "Undercuts",
            AlgebraicClass::Defeating,
            ["undercuts"],
            Some("undercuts"),
        ),
        EdgeTypeSpec::new(
            "Contradicts",
            AlgebraicClass::Defeating,
            ["CONTRADICTS", "contradicts", "ATTACKS", "attacks"],
            Some("contradicts"),
        ),
        EdgeTypeSpec::new(
            "Tension",
            AlgebraicClass::Defeating,
            ["tension"],
            Some("tension"),
        ),
        EdgeTypeSpec::new(
            "SameEClass",
            AlgebraicClass::Equivalence,
            Vec::<String>::new(),
            None,
        ),
        EdgeTypeSpec::new(
            "HasEpistemicShadow",
            AlgebraicClass::Structural,
            Vec::<String>::new(),
            None,
        ),
    ]
}

pub const CALLS_SYMBOL_EDGE: &str = "CALLS_SYMBOL";
pub const DEPENDS_ON_SYMBOL_EDGE: &str = "DEPENDS_ON_SYMBOL";
pub const MENTIONS_LITERAL_EDGE: &str = "MENTIONS_LITERAL";
pub const RENDEZVOUS_EDGE: &str = "RENDEZVOUS";
pub const SCIP_DEFINES_EDGE: &str = "SCIP_DEFINES";
pub const SCIP_REFERENCES_EDGE: &str = "SCIP_REFERENCES";
pub const SCIP_JOINS_CODE_SYMBOL_EDGE: &str = "SCIP_JOINS_CODE_SYMBOL";
pub const CONTAINS_EDGE: &str = "CONTAINS";
pub const WRITES_FIELD_EDGE: &str = "WRITES_FIELD";
pub const READS_FIELD_EDGE: &str = "READS_FIELD";
pub const CHANGED_FILE_EDGE: &str = "CHANGED_FILE";

/// Ontology rows for the code graph.
///
/// Symbol relationships and precise-index joins are positive evidence. File
/// containment is structural and therefore explainable without minting a
/// provenance variable. `RENDEZVOUS` is the one symmetric code relation.
pub fn code_edge_specs() -> Vec<EdgeTypeSpec> {
    vec![
        EdgeTypeSpec::new(
            CALLS_SYMBOL_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        ),
        EdgeTypeSpec::new(
            DEPENDS_ON_SYMBOL_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        ),
        EdgeTypeSpec::new(
            MENTIONS_LITERAL_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        )
        .with_weight_property("role_hint"),
        EdgeTypeSpec::new(
            RENDEZVOUS_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        )
        .with_direction(EdgeDirection::Symmetric),
        EdgeTypeSpec::new(
            SCIP_DEFINES_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        ),
        EdgeTypeSpec::new(
            SCIP_REFERENCES_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        ),
        EdgeTypeSpec::new(
            SCIP_JOINS_CODE_SYMBOL_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        ),
        EdgeTypeSpec::new(
            CONTAINS_EDGE,
            AlgebraicClass::Structural,
            Vec::<String>::new(),
            None,
        ),
        EdgeTypeSpec::new(
            WRITES_FIELD_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        ),
        EdgeTypeSpec::new(
            READS_FIELD_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        ),
        EdgeTypeSpec::new(
            CHANGED_FILE_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        ),
    ]
}

/// Process-wide ontology that can classify derivation, epistemic, and code
/// graph edges without granting unknown relations a default weight.
pub fn code_graph_ontology() -> &'static EdgeOntology {
    static REGISTRY: OnceLock<EdgeOntology> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut specs = default_specs();
        specs.extend(code_edge_specs());
        EdgeOntology::from_specs(specs).expect("code graph ontology is well formed")
    })
}

/// The process-wide default registry.
pub fn default_registry() -> &'static EdgeOntology {
    static REGISTRY: OnceLock<EdgeOntology> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        EdgeOntology::from_specs(default_specs()).expect("default edge ontology is well formed")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_resolves_every_documented_token() {
        let ontology = default_registry();
        let expected = [
            ("Supports", AlgebraicClass::MonotonePositive),
            ("supports", AlgebraicClass::MonotonePositive),
            ("SUPPORTS", AlgebraicClass::MonotonePositive),
            ("Cites", AlgebraicClass::MonotonePositive),
            ("CITES", AlgebraicClass::MonotonePositive),
            ("cites", AlgebraicClass::MonotonePositive),
            ("Derives", AlgebraicClass::MonotonePositive),
            ("DERIVED_FROM", AlgebraicClass::MonotonePositive),
            ("derived_from", AlgebraicClass::MonotonePositive),
            ("derives", AlgebraicClass::MonotonePositive),
            ("Undercuts", AlgebraicClass::Defeating),
            ("undercuts", AlgebraicClass::Defeating),
            ("Contradicts", AlgebraicClass::Defeating),
            ("CONTRADICTS", AlgebraicClass::Defeating),
            ("contradicts", AlgebraicClass::Defeating),
            ("ATTACKS", AlgebraicClass::Defeating),
            ("attacks", AlgebraicClass::Defeating),
            ("Tension", AlgebraicClass::Defeating),
            ("tension", AlgebraicClass::Defeating),
            ("SameEClass", AlgebraicClass::Equivalence),
            ("HasEpistemicShadow", AlgebraicClass::Structural),
        ];
        for (token, class) in expected {
            assert_eq!(
                ontology.class_of(token),
                Some(class),
                "token {token} did not resolve to {class:?}"
            );
        }
    }

    #[test]
    fn resolution_is_case_insensitive_and_trims() {
        let ontology = default_registry();
        assert_eq!(
            ontology.class_of("  sUpPoRtS  "),
            Some(AlgebraicClass::MonotonePositive)
        );
        assert_eq!(
            ontology.class_of("hasepistemicshadow"),
            Some(AlgebraicClass::Structural)
        );
    }

    #[test]
    fn unknown_edge_types_resolve_to_none_not_a_default() {
        let ontology = default_registry();
        for token in ["", "  ", "mentions", "RELATED_TO", "supports_maybe"] {
            assert_eq!(ontology.resolve(token), None, "token {token:?} resolved");
            assert_eq!(ontology.class_of(token), None, "token {token:?} classified");
        }
    }

    #[test]
    fn epistemic_type_tokens_are_present_exactly_for_valued_rows() {
        for spec in default_registry().specs() {
            if spec.class.is_valued() {
                assert!(
                    spec.epistemic_type.is_some(),
                    "{} is valued but carries no epistemic type",
                    spec.canonical
                );
            } else {
                assert_eq!(
                    spec.epistemic_type, None,
                    "{} is not valued but carries an epistemic type",
                    spec.canonical
                );
            }
        }
    }

    #[test]
    fn class_filters_return_declaration_order() {
        let ontology = default_registry();
        assert_eq!(
            ontology.canonical_names_in_class(AlgebraicClass::MonotonePositive),
            vec!["Supports", "Cites", "Derives"]
        );
        assert_eq!(
            ontology.canonical_names_in_class(AlgebraicClass::Defeating),
            vec!["Undercuts", "Contradicts", "Tension"]
        );
        // Case variants of a token already emitted are folded away: resolution
        // is case-insensitive, so one spelling stands for all of them.
        assert_eq!(
            ontology.tokens_in_class(AlgebraicClass::Defeating),
            vec!["Undercuts", "Contradicts", "ATTACKS", "Tension"]
        );
        for token in [
            "undercuts",
            "CONTRADICTS",
            "contradicts",
            "attacks",
            "tension",
        ] {
            assert_eq!(ontology.class_of(token), Some(AlgebraicClass::Defeating));
        }
    }

    #[test]
    fn duplicate_canonical_names_are_rejected_with_both_tokens() {
        let err = EdgeOntology::from_specs(vec![
            EdgeTypeSpec::new("Supports", AlgebraicClass::MonotonePositive, ["s"], None),
            EdgeTypeSpec::new("SUPPORTS", AlgebraicClass::Defeating, ["t"], None),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            OntologyError::DuplicateCanonical {
                tokens: vec!["Supports".to_string(), "SUPPORTS".to_string()]
            }
        );
    }

    #[test]
    fn alias_collisions_across_rows_are_rejected_with_all_tokens() {
        let err = EdgeOntology::from_specs(vec![
            EdgeTypeSpec::new(
                "Supports",
                AlgebraicClass::MonotonePositive,
                ["backs"],
                None,
            ),
            EdgeTypeSpec::new("Undercuts", AlgebraicClass::Defeating, ["BACKS"], None),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            OntologyError::AliasCollision {
                tokens: vec![
                    "BACKS".to_string(),
                    "Undercuts".to_string(),
                    "Supports".to_string()
                ]
            }
        );
    }

    #[test]
    fn alias_colliding_with_another_canonical_is_rejected() {
        let err = EdgeOntology::from_specs(vec![
            EdgeTypeSpec::new(
                "Supports",
                AlgebraicClass::MonotonePositive,
                Vec::<String>::new(),
                None,
            ),
            EdgeTypeSpec::new("Undercuts", AlgebraicClass::Defeating, ["supports"], None),
        ])
        .unwrap_err();
        assert!(matches!(err, OntologyError::AliasCollision { .. }));
    }

    #[test]
    fn alias_that_case_folds_onto_its_own_canonical_is_idempotent() {
        let ontology = EdgeOntology::from_specs(vec![EdgeTypeSpec::new(
            "Supports",
            AlgebraicClass::MonotonePositive,
            ["supports", "SUPPORTS", "supports"],
            Some("supports"),
        )])
        .expect("self-folding aliases are not collisions");
        assert_eq!(
            ontology.class_of("SuPpOrTs"),
            Some(AlgebraicClass::MonotonePositive)
        );
    }

    #[test]
    fn empty_canonical_names_are_rejected() {
        let err = EdgeOntology::from_specs(vec![EdgeTypeSpec::new(
            "   ",
            AlgebraicClass::Structural,
            Vec::<String>::new(),
            None,
        )])
        .unwrap_err();
        assert_eq!(err, OntologyError::EmptyCanonical);
    }

    #[test]
    fn a_spec_row_without_a_class_fails_to_deserialize_rather_than_defaulting() {
        let err = serde_json::from_str::<EdgeTypeSpec>(r#"{"canonical":"Mentions"}"#)
            .expect_err("class must be declared");
        assert!(err.to_string().contains("class"), "unexpected error: {err}");
    }

    #[test]
    fn specs_round_trip_through_serde() {
        let specs = default_specs();
        let json = serde_json::to_string(&specs).expect("serialize");
        let restored: Vec<EdgeTypeSpec> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, specs);
        assert_eq!(
            serde_json::to_value(AlgebraicClass::MonotonePositive).unwrap(),
            serde_json::json!("monotone_positive")
        );
    }

    #[test]
    fn code_edges_declare_direction_class_and_weight_source() {
        let ontology = code_graph_ontology();
        let rendezvous = ontology.resolve(RENDEZVOUS_EDGE).expect("rendezvous");
        assert_eq!(rendezvous.class, AlgebraicClass::MonotonePositive);
        assert_eq!(rendezvous.direction, EdgeDirection::Symmetric);

        let mentions = ontology
            .resolve(MENTIONS_LITERAL_EDGE)
            .expect("mentions literal");
        assert_eq!(mentions.direction, EdgeDirection::Directed);
        assert_eq!(mentions.weight_property.as_deref(), Some("role_hint"));

        for edge_type in [
            CALLS_SYMBOL_EDGE,
            DEPENDS_ON_SYMBOL_EDGE,
            MENTIONS_LITERAL_EDGE,
            RENDEZVOUS_EDGE,
            SCIP_DEFINES_EDGE,
            SCIP_REFERENCES_EDGE,
            SCIP_JOINS_CODE_SYMBOL_EDGE,
            CONTAINS_EDGE,
            WRITES_FIELD_EDGE,
            READS_FIELD_EDGE,
            CHANGED_FILE_EDGE,
        ] {
            assert!(ontology.resolve(edge_type).is_some(), "missing {edge_type}");
        }
    }
}
