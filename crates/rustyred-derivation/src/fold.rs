//! Recursive folds over a derivation graph.
//!
//! Two entry points over the same traversal: [`fold`] accumulates a scalar in
//! one semiring, [`fold_dual`] accumulates the free dual-indeterminate
//! polynomial that every semiring can then read out. Their node semantics are
//! identical, which is what makes the universality parity in
//! [`crate::interpret`](mod@crate::interpret) a real check rather than a restatement.

use std::collections::{HashMap, HashSet};

use crate::dual::{DualPolynomial, EdgeToken, Generator, Polarity, MAX_DUAL_MONOMIALS};
use crate::ontology::{AlgebraicClass, EdgeOntology, EdgeTypeSpec};
use crate::semiring::Semiring;
use crate::store::{
    DerivationStore, ANNOTATION_PROPERTY, ASSUMPTION_LABEL, DERIVATION_ANCESTOR_EDGE,
    DERIVATION_ASSUMES_EDGE, DERIVATION_LABEL, MAX_DERIVATION_DEPTH,
};

/// Accumulate the semiring fold for `node_id` in `store`.
pub fn fold<S: Semiring>(store: &impl DerivationStore, node_id: &str) -> S::Element {
    let mut memo: HashMap<String, S::Element> = HashMap::new();
    let mut in_progress: HashSet<String> = HashSet::new();
    fold_recursive::<S>(store, node_id, 0, &mut memo, &mut in_progress)
}

fn fold_recursive<S: Semiring>(
    store: &impl DerivationStore,
    node_id: &str,
    depth: usize,
    memo: &mut HashMap<String, S::Element>,
    in_progress: &mut HashSet<String>,
) -> S::Element {
    if in_progress.contains(node_id) {
        return S::zero();
    }
    if !S::is_absorptive() && depth > MAX_DERIVATION_DEPTH {
        return S::zero();
    }
    if let Some(cached) = memo.get(node_id) {
        return cached.clone();
    }

    let labels = match store.node_labels(node_id) {
        Some(labels) => labels,
        None => {
            memo.insert(node_id.to_string(), S::zero());
            return S::zero();
        }
    };

    if labels.iter().any(|l| l == ASSUMPTION_LABEL) {
        let result = S::one();
        memo.insert(node_id.to_string(), result.clone());
        return result;
    }
    if !labels.iter().any(|l| l == DERIVATION_LABEL) {
        memo.insert(node_id.to_string(), S::zero());
        return S::zero();
    }

    let incoming_edges = gather_incoming_edges(store, node_id);
    if incoming_edges.is_empty() {
        let result = S::one();
        memo.insert(node_id.to_string(), result.clone());
        return result;
    }

    in_progress.insert(node_id.to_string());
    let mut acc = S::zero();
    for (predecessor_id, annotation) in &incoming_edges {
        let pred_value = fold_recursive::<S>(store, predecessor_id, depth + 1, memo, in_progress);
        let edge_weight = S::from_annotation(*annotation);
        let edge_value = S::product(&edge_weight, &pred_value);
        acc = S::sum(&acc, &edge_value);
    }
    in_progress.remove(node_id);
    memo.insert(node_id.to_string(), acc.clone());
    acc
}

fn gather_incoming_edges(store: &impl DerivationStore, node_id: &str) -> Vec<(String, f64)> {
    let mut results = Vec::new();
    for hit in store.neighbors_in(node_id, DERIVATION_ANCESTOR_EDGE) {
        let annotation = store
            .edge_annotation(&hit.edge_id, ANNOTATION_PROPERTY)
            .unwrap_or(1.0);
        results.push((hit.node_id, annotation));
    }
    for hit in store.neighbors_in(node_id, DERIVATION_ASSUMES_EDGE) {
        let annotation = store
            .edge_annotation(&hit.edge_id, ANNOTATION_PROPERTY)
            .unwrap_or(1.0);
        results.push((hit.node_id, annotation));
    }
    results
}

// ---------------------------------------------------------------------------
// Fold into the free layer
// ---------------------------------------------------------------------------

/// The number of node expansions one [`fold_dual`] may perform.
///
/// [`MAX_DERIVATION_DEPTH`] and [`MAX_DUAL_MONOMIALS`] both bound the *result*,
/// and neither bounds the *work*. A cyclic component with no reachable
/// assumption produces zero polynomials, so the term budget never trips, and a
/// component of a dozen nodes sits far below the depth bound while holding
/// hundreds of thousands of simple paths. Since a cycle also disables
/// memoization for everything above it, that path count is enumerated. This
/// bounds the traversal itself, so such a graph refuses instead of hanging.
///
/// An acyclic graph memoizes, so it expands each node about once and never
/// approaches this.
pub const MAX_DUAL_EXPANSIONS: usize = 100_000;

/// A fold that could not mint a generator without guessing.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum DualFoldError {
    #[error("edge {edge_id} carries no edge type, so its polarity cannot be determined")]
    UntypedEdge { edge_id: String },
    #[error(
        "edge {edge_id} has edge type {edge_type:?}, which the edge ontology does not classify"
    )]
    UnclassifiedEdge { edge_id: String, edge_type: String },
    #[error("store cannot enumerate incoming edge types for code subject {node_id}")]
    IncomingEdgeTypesUnavailable { node_id: String },
    /// The traversal ran past [`MAX_DERIVATION_DEPTH`].
    ///
    /// The scalar fold answers zero here, but it only applies the bound to
    /// non-absorptive semirings; Boolean, Tropical, Viterbi, and Corroboration
    /// keep descending and still reach the assumption. A zero polynomial reads
    /// as "no derivation exists" under every one of those, which would be false
    /// and would break the universality parity this layer promises. Refusing
    /// says the fold does not know, which is true.
    #[error("derivation depth exceeded {limit} at node {node_id}")]
    DepthExceeded { node_id: String, limit: usize },
    /// The polynomial outgrew [`MAX_DUAL_MONOMIALS`].
    #[error("dual polynomial for node {node_id} reached {monomials} monomials, above the bound of {limit}")]
    TermBudgetExceeded {
        node_id: String,
        monomials: usize,
        limit: usize,
    },
    /// The traversal outgrew [`MAX_DUAL_EXPANSIONS`].
    ///
    /// Cycles disable memoization, so a dense cyclic component enumerates its
    /// simple paths. When no assumption is reachable every polynomial stays
    /// zero, so neither the depth bound nor the term budget ever fires. This
    /// bounds the work directly.
    #[error("dual fold expanded {expansions} nodes, above the bound of {limit}")]
    ExpansionBudgetExceeded { expansions: usize, limit: usize },
}

/// Fold an explicit examined code-edge set into provenance variables.
///
/// Code facts are set-query results, not recursive derivation DAGs. Their
/// provenance is therefore the sum of the exact graph edges the query
/// examined. Structural and equivalence edges remain explainable but do not
/// mint semiring variables, matching the ontology contract.
pub fn fold_code_edges<'a>(
    store: &impl DerivationStore,
    edge_ids: impl IntoIterator<Item = &'a str>,
    ontology: &EdgeOntology,
) -> Result<DualPolynomial, DualFoldError> {
    let mut polynomial = DualPolynomial::zero();
    let mut seen = HashSet::new();
    for edge_id in edge_ids {
        if !seen.insert(edge_id.to_string()) {
            continue;
        }
        let edge_type = store
            .edge_type(edge_id)
            .ok_or_else(|| DualFoldError::UntypedEdge {
                edge_id: edge_id.to_string(),
            })?;
        let spec = ontology
            .resolve(&edge_type)
            .ok_or_else(|| DualFoldError::UnclassifiedEdge {
                edge_id: edge_id.to_string(),
                edge_type: edge_type.clone(),
            })?;
        let Some(polarity) = Polarity::for_class(spec.class) else {
            continue;
        };
        polynomial = polynomial.plus(&DualPolynomial::generator(Generator::new(
            edge_id,
            polarity,
            &spec.canonical,
        )));
    }
    Ok(polynomial)
}

/// Fold every incoming edge of a code subject.
///
/// Enumeration is mandatory: falling back to the ontology's known spellings
/// would make an unknown stored edge invisible, violating the refusal rule.
pub fn fold_code_incoming(
    store: &impl DerivationStore,
    node_id: &str,
    ontology: &EdgeOntology,
) -> Result<DualPolynomial, DualFoldError> {
    let mut edge_types = store.incoming_edge_types(node_id).ok_or_else(|| {
        DualFoldError::IncomingEdgeTypesUnavailable {
            node_id: node_id.to_string(),
        }
    })?;
    edge_types.sort();
    edge_types.dedup();

    let mut edge_ids = Vec::new();
    for edge_type in edge_types {
        let mut hits = store.neighbors_in(node_id, &edge_type);
        hits.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
        edge_ids.extend(hits.into_iter().map(|hit| hit.edge_id));
    }
    fold_code_edges(store, edge_ids.iter().map(String::as_str), ontology)
}

/// The ontology for folding *derivation* DAGs, as distinct from the epistemic
/// graph.
///
/// [`crate::store`] owns `derivation_ancestor` and `derivation_assumes`, so
/// this crate is the right place to declare their algebraic class. Both compose
/// monotonically — an ancestor edge is a positive premise link — so both are
/// [`AlgebraicClass::MonotonePositive`]. Rows from
/// [`crate::ontology::default_specs`] are carried along unchanged, so an
/// epistemic edge appearing inside a derivation DAG still resolves.
pub fn derivation_dag_ontology() -> &'static EdgeOntology {
    static REGISTRY: std::sync::OnceLock<EdgeOntology> = std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut specs = crate::ontology::default_specs();
        specs.push(EdgeTypeSpec::new(
            DERIVATION_ANCESTOR_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        ));
        specs.push(EdgeTypeSpec::new(
            DERIVATION_ASSUMES_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        ));
        EdgeOntology::from_specs(specs).expect("derivation DAG ontology is well formed")
    })
}

/// Every edge-type spelling the dual fold traverses.
///
/// A graph neighbor query matches an edge type exactly, so asking only for
/// `derivation_ancestor` and `derivation_assumes` can return only those two,
/// and both are `MonotonePositive`. That made every `Defeating` row unreachable
/// and the fold positive-only in production, which defeats the point of the
/// dual layer: the free layer exists so that defeat is foldable rather than
/// routed around. The traversal therefore asks for every spelling of every
/// valued class the ontology declares.
///
/// Only `Defeating` spellings are added. Positive composition already flows
/// through the two derivation relations, so pulling in `Supports`, `Cites`, or
/// `Derives` as well would let an incidental graph link on a derivation node
/// count as an alternative proof. The scalar [`fold`] queries only the two
/// derivation relations, so that would inflate the counting and probability
/// readouts and break the universality parity on any mixed-purpose graph.
///
/// The asymmetry that remains is deliberate and parity-safe: the scalar fold
/// omits defeating edges because no semiring can express attack, and a
/// positive-only DAG carries none, so both folds see the same set wherever
/// parity is claimed.
fn traversal_edge_types(ontology: &EdgeOntology) -> Vec<String> {
    let mut types = vec![
        DERIVATION_ANCESTOR_EDGE.to_string(),
        DERIVATION_ASSUMES_EDGE.to_string(),
    ];
    let mut seen: HashSet<String> = types.iter().cloned().collect();
    for spelling in ontology.edge_type_spellings_in_class(AlgebraicClass::Defeating) {
        if seen.insert(spelling.clone()) {
            types.push(spelling);
        }
    }
    types
}

/// The edge types to fetch at one node.
///
/// [`traversal_edge_types`] can only name the spellings the registry lists,
/// while [`DerivationStore::neighbors_in`] matches exactly and
/// [`EdgeOntology::resolve`] is case-insensitive. A stored `UNDERCUTS` is
/// therefore `Defeating` on inspection but unreachable by a traversal built
/// from the registry alone, so the attack vanishes and the readout can call an
/// attacked node supported. A store that can enumerate its own incoming edge
/// types is asked instead, and the ontology classifies what it reports.
///
/// The two derivation relations stay exact-match rather than
/// classify-by-ontology. The scalar [`fold`] queries those two literal
/// constants, so accepting case variants here would fetch edges the scalar
/// fold cannot see and break the universality parity that
/// [`crate::interpret`](mod@crate::interpret) asserts. Only `Defeating` is widened, because the
/// scalar fold carries no defeating edges at all and parity is unaffected.
/// Does `node_id` sit in a mutual defeating pair?
///
/// Mirrors `contradiction_cycles` in the structural pass, which pairs an attack
/// with its reverse and marks both ends. A defeating self-loop satisfies that
/// same test and is included here for the same reason.
///
/// Zero-weight attacks are skipped, matching the edge collection below: an
/// attack asserting nothing cannot be half of a standoff.
/// Per-fold caches for the defeating-cycle question.
///
/// `membership` holds the answer per node. `sources` holds each node's
/// defeating predecessors, and it is the one that matters: caching only the
/// answer still left every node running its own walk and re-probing the store
/// for the same edges, which on a six-by-six defeating DAG cost 452 probes for
/// 31 nodes.
#[derive(Default)]
struct DefeatCaches {
    membership: HashMap<String, bool>,
    sources: HashMap<String, Vec<String>>,
}

fn in_defeating_cycle<S: DerivationStore + ?Sized>(
    store: &S,
    node_id: &str,
    ontology: &EdgeOntology,
    expansions: &mut usize,
    caches: &mut DefeatCaches,
) -> Result<bool, DualFoldError> {
    // Answered once per node per fold. Every node reached by the recursion asks
    // this question, and on a dense defeating graph with shared predecessors
    // each answer re-walked most of the same subgraph, so the probes grew
    // quadratically. That work also ran outside MAX_DUAL_EXPANSIONS, so it
    // could outrun the fold's own bounds before any budget fired. It is cached
    // here and charged below.
    if let Some(known) = caches.membership.get(node_id) {
        return Ok(*known);
    }
    // Walk backwards along defeating edges. Arriving back at the start means
    // the start sits on a directed defeating cycle.
    //
    // This began as a single reverse-edge test, mirroring `contradiction_cycles`
    // in the structural pass, which pairs an attack with its reverse. That
    // caught the self-loop and the mutual pair and nothing longer, and the
    // one-level rule then answered a longer ring by its parity: a three-ring
    // suppressed every attack and read `In`, a four-ring left one and read
    // `Out`. A standing on a ring is not a function of whether the ring is
    // odd.
    //
    // `seen` bounds the walk to the defeating subgraph reachable from
    // `node_id`, so each call is linear in that subgraph rather than
    // unbounded.
    // Breadth first, one hop per round, so the bound counts path length rather
    // than how many nodes happened to be visited.
    //
    // A depth-first walk bounded on the size of the visited set was
    // order-dependent: a node on a short cycle that also reaches 256 other
    // defeating predecessors could exhaust the budget before the stack popped
    // the anchor, and whether that happened turned on neighbour ordering. The
    // question asked here, "is the anchor reachable from itself within
    // MAX_DERIVATION_DEPTH hops", is a property of the graph alone, so the
    // answer must not depend on traversal order.
    //
    // Stopping at the same bound the fold enforces keeps
    // MAX_DERIVATION_DEPTH meaning one thing. This prepass answers before the
    // recursion runs, so an unbounded walk would settle a ring the fold itself
    // would refuse. Reporting "not a cycle" hands the node to the ordinary
    // path, which walks it and raises DepthExceeded.
    let mut seen: HashSet<String> = HashSet::new();
    let mut frontier: Vec<String> = cached_sources(store, node_id, ontology, caches);
    let mut answer = false;
    'search: for _ in 0..MAX_DERIVATION_DEPTH {
        if frontier.is_empty() {
            break;
        }
        if frontier.iter().any(|candidate| candidate == node_id) {
            answer = true;
            break;
        }
        let mut next = Vec::new();
        for current in frontier {
            if seen.insert(current.clone()) {
                *expansions += 1;
                if *expansions > MAX_DUAL_EXPANSIONS {
                    return Err(DualFoldError::ExpansionBudgetExceeded {
                        expansions: *expansions,
                        limit: MAX_DUAL_EXPANSIONS,
                    });
                }
                next.extend(cached_sources(store, &current, ontology, caches));
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break 'search;
        }
    }
    caches.membership.insert(node_id.to_string(), answer);
    Ok(answer)
}

/// `defeating_sources` for one node, fetched from the store at most once per
/// fold.
fn cached_sources<S: DerivationStore + ?Sized>(
    store: &S,
    node_id: &str,
    ontology: &EdgeOntology,
    caches: &mut DefeatCaches,
) -> Vec<String> {
    if let Some(known) = caches.sources.get(node_id) {
        return known.clone();
    }
    let sources = defeating_sources(store, node_id, ontology);
    caches.sources.insert(node_id.to_string(), sources.clone());
    sources
}

/// The nodes whose defeating edges reach `node_id`, with weightless attacks
/// dropped.
fn defeating_sources<S: DerivationStore + ?Sized>(
    store: &S,
    node_id: &str,
    ontology: &EdgeOntology,
) -> Vec<String> {
    let mut sources = Vec::new();
    for edge_type in node_traversal_edge_types(store, node_id, ontology) {
        if ontology.class_of(&edge_type) != Some(AlgebraicClass::Defeating) {
            continue;
        }
        for hit in store.neighbors_in(node_id, &edge_type) {
            if store.edge_annotation(&hit.edge_id, ANNOTATION_PROPERTY) == Some(0.0) {
                continue;
            }
            sources.push(hit.node_id);
        }
    }
    sources
}

fn node_traversal_edge_types<S: DerivationStore + ?Sized>(
    store: &S,
    node_id: &str,
    ontology: &EdgeOntology,
) -> Vec<String> {
    let Some(stored) = store.incoming_edge_types(node_id) else {
        return traversal_edge_types(ontology);
    };
    let mut types = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for edge_type in stored {
        let wanted = edge_type == DERIVATION_ANCESTOR_EDGE
            || edge_type == DERIVATION_ASSUMES_EDGE
            || ontology.class_of(&edge_type) == Some(AlgebraicClass::Defeating);
        if wanted && seen.insert(edge_type.clone()) {
            types.push(edge_type);
        }
    }
    types
}

/// Fold a derivation DAG once into a [`DualPolynomial`].
///
/// Alternatives combine through [`DualPolynomial::plus`], conjunctive premises
/// through [`DualPolynomial::times`]. Each edge mints its [`Generator`] with the
/// polarity `ontology.class_of` gives its edge type; `Equivalence` and
/// `Structural` edges mint nothing and pass through as the multiplicative
/// identity.
///
/// Node semantics mirror [`fold`] exactly, which is what makes the universality
/// parity in [`crate::interpret`](mod@crate::interpret) meaningful: assumptions and premise-free
/// derivations are `one()`, missing and non-derivation nodes are `zero()`, and
/// a cycle contributes `zero()`.
///
/// Three conditions are refusals rather than values, because in each case a
/// returned polynomial would assert something false:
///
/// - an edge type the ontology does not classify, since default trust is never
///   granted silently;
/// - traversal past [`MAX_DERIVATION_DEPTH`], since a zero polynomial would
///   claim no derivation exists to the four absorptive semirings whose scalar
///   fold has no depth bound at all;
/// - a polynomial past [`MAX_DUAL_MONOMIALS`], since path expansion is
///   multiplicative and materializing it would exhaust memory.
pub fn fold_dual(
    store: &impl DerivationStore,
    node_id: &str,
    ontology: &EdgeOntology,
) -> Result<DualPolynomial, DualFoldError> {
    let mut memo: HashMap<String, MemoEntry> = HashMap::new();
    let mut in_progress: HashSet<String> = HashSet::new();
    let mut expansions = 0usize;
    let mut caches = DefeatCaches::default();
    fold_dual_recursive(
        store,
        node_id,
        0,
        ontology,
        &mut memo,
        &mut in_progress,
        &mut expansions,
        &mut caches,
    )
    .map(|outcome| outcome.entry.polynomial)
}

/// A memoized subtree: its polynomial, and how far below the node the traversal
/// reached to build it.
///
/// The height is what makes the depth refusal independent of traversal order.
/// The memo is keyed by node alone, so a shared suffix cached through a short
/// branch would otherwise be handed back to a longer branch whose complete path
/// runs past [`MAX_DERIVATION_DEPTH`], and whether an over-depth graph was
/// accepted or refused would depend on the order the store returned edges in.
/// Recording the height lets a cache hit at depth `d` answer the same question
/// a fresh traversal would: does `d + height` exceed the bound.
#[derive(Clone)]
struct MemoEntry {
    polynomial: DualPolynomial,
    height: usize,
}

impl MemoEntry {
    fn leaf(polynomial: DualPolynomial) -> Self {
        Self {
            polynomial,
            height: 0,
        }
    }
}

/// One node's fold, plus whether a cycle was cut anywhere beneath it.
///
/// A cycle is cut by answering zero for a node already on the stack, and that
/// zero holds only while that ancestor is being expanded. Memoization cannot
/// be reconciled with simple-path semantics once any cycle is in play, and it
/// fails in both directions. Caching the truncated value loses derivations:
/// for `P -> A -> B`, `B -> A`, with both `A -> T` and `B -> T`, expanding `A`
/// first cached `B` as zero and dropped `P -> A -> B -> T`. Caching the
/// untruncated value invents them: expanding `B` first cached `B` as
/// `P -> A -> B`, which `A` then reused to build `P -> A -> B -> A -> T`, a
/// walk that revisits `A` and is no derivation at all.
///
/// So a value is cacheable only when nothing under it was cut. Acyclic graphs,
/// the ordinary case, are unaffected and memoize exactly as before. Inside and
/// above a cycle the fold recomputes in each context, which is bounded by
/// [`MAX_DERIVATION_DEPTH`] and the [`MAX_DUAL_MONOMIALS`] term budget: a
/// pathological graph returns a typed refusal rather than running away.
struct FoldOutcome {
    entry: MemoEntry,
    cycle_truncated: bool,
}

impl FoldOutcome {
    /// A value that stands on its own, with no cycle cut beneath it.
    fn settled(entry: MemoEntry) -> Self {
        Self {
            entry,
            cycle_truncated: false,
        }
    }
}

fn fold_dual_recursive(
    store: &impl DerivationStore,
    node_id: &str,
    depth: usize,
    ontology: &EdgeOntology,
    memo: &mut HashMap<String, MemoEntry>,
    in_progress: &mut HashSet<String>,
    expansions: &mut usize,
    caches: &mut DefeatCaches,
) -> Result<FoldOutcome, DualFoldError> {
    *expansions += 1;
    if *expansions > MAX_DUAL_EXPANSIONS {
        return Err(DualFoldError::ExpansionBudgetExceeded {
            expansions: *expansions,
            limit: MAX_DUAL_EXPANSIONS,
        });
    }
    // Depth first. The bound is a refusal, and a path that closes a cycle has
    // still walked that far: checking `in_progress` ahead of it let a cycle
    // longer than the limit return a truncated zero instead of refusing, so
    // the same traversal was bounded on an acyclic path and unbounded on a
    // cyclic one.
    if depth > MAX_DERIVATION_DEPTH {
        return Err(DualFoldError::DepthExceeded {
            node_id: node_id.to_string(),
            limit: MAX_DERIVATION_DEPTH,
        });
    }
    if in_progress.contains(node_id) {
        // Cut the cycle, and name the ancestor we cut it at so no caller
        // downstream of this zero mistakes it for a settled value.
        return Ok(FoldOutcome {
            entry: MemoEntry::leaf(DualPolynomial::zero()),
            cycle_truncated: true,
        });
    }
    if let Some(cached) = memo.get(node_id) {
        // Answer the question a fresh traversal of this subtree would answer.
        if depth + cached.height > MAX_DERIVATION_DEPTH {
            return Err(DualFoldError::DepthExceeded {
                node_id: node_id.to_string(),
                limit: MAX_DERIVATION_DEPTH,
            });
        }
        return Ok(FoldOutcome::settled(cached.clone()));
    }

    let Some(labels) = store.node_labels(node_id) else {
        let entry = MemoEntry::leaf(DualPolynomial::zero());
        memo.insert(node_id.to_string(), entry.clone());
        return Ok(FoldOutcome::settled(entry));
    };

    let is_assumption = labels.iter().any(|label| label == ASSUMPTION_LABEL);
    if !is_assumption && !labels.iter().any(|label| label == DERIVATION_LABEL) {
        let entry = MemoEntry::leaf(DualPolynomial::zero());
        memo.insert(node_id.to_string(), entry.clone());
        return Ok(FoldOutcome::settled(entry));
    }

    // A node in a mutual defeating pair contributes zero on both sides and so
    // reads `Undecided`, which is what `contradiction_cycles` in the structural
    // pass answers and what `epistemic_node_polynomial` already does for the
    // same shape. Without this the two attacks resolved against each other by
    // traversal order: folding A cut the return leg B -> A, so B's own
    // defeating fragment looked empty, B's attack on A was allowed to land and
    // A read `Out`. The doc comment on `epistemic_node_polynomial` already
    // claimed this fold used the cycle convention; now it does.
    if in_defeating_cycle(store, node_id, ontology, expansions, caches)? {
        let entry = MemoEntry::leaf(DualPolynomial::zero());
        memo.insert(node_id.to_string(), entry.clone());
        return Ok(FoldOutcome::settled(entry));
    }

    let mut edges: Vec<crate::store::DerivationNeighbor> = Vec::new();
    let mut seen_edges: HashSet<String> = HashSet::new();
    for edge_type in node_traversal_edge_types(store, node_id, ontology) {
        // An assumption has no derivation to extend, so only its attackers
        // matter. It used to return the unit before edge collection ran at
        // all, which meant an attacked assumption minted no negative
        // generator and read `In`.
        let defeating_type = ontology.class_of(&edge_type) == Some(AlgebraicClass::Defeating);
        if is_assumption && !defeating_type {
            continue;
        }
        for hit in store.neighbors_in(node_id, &edge_type) {
            // A zero-weight attack asserts nothing, so it does not belong in
            // the polynomial at all. `epistemic_node_polynomial` already drops
            // such relations for the same reason. Admitting one put a
            // generator carrying no weight on the against side, and once a
            // defeated attacker stopped carrying its attack, that weightless
            // generator was enough to suppress a live attack one hop further
            // on: a zero-weight `C -> B` made B look defeated and silently
            // cancelled B's real attack on A.
            //
            // Exactly zero only, matching `normalized_confidence`, which reads
            // a negative or non-finite annotation as fully weighted rather
            // than as an absent assertion. An edge carrying no annotation at
            // all is unweighted, not weightless, and stays.
            if defeating_type
                && store.edge_annotation(&hit.edge_id, ANNOTATION_PROPERTY) == Some(0.0)
            {
                continue;
            }
            if seen_edges.insert(hit.edge_id.clone()) {
                edges.push(hit);
            }
        }
    }
    if edges.is_empty() {
        let entry = MemoEntry::leaf(DualPolynomial::one());
        memo.insert(node_id.to_string(), entry.clone());
        return Ok(FoldOutcome::settled(entry));
    }

    // Mint every edge's generator once, before deciding the seed. The seed
    // turns on whether any *premise* reaches this node, not on whether any
    // edge does. Keying it off `edges.is_empty()` cost a premise-free
    // derivation its unit the moment an attacker appeared: collecting the
    // attack made `edges` nonempty, the node is not an assumption, so the
    // accumulator started at zero and the node reported no support at all
    // purely because something attacked it. The scalar fold answers `one()`
    // for that same premise-free derivation, so that was a parity break.
    //
    // Minting here is deliberately tolerant of a mint error. The loop below
    // mints again and refuses there, which keeps the refusal naming the same
    // edge it always named: those refusals are asserted by edge id, and the
    // order they surface in is observable behaviour that no finding asked to
    // change.
    let has_premise = edges
        .iter()
        .any(|hit| match mint_generator(store, &hit.edge_id, ontology) {
            Ok(generator) => {
                generator.is_none_or(|generator| generator.polarity == Polarity::Positive)
            }
            Err(_) => false,
        });

    in_progress.insert(node_id.to_string());
    // A node nothing derives stands on its own, the way an assumption does.
    // A node something derives is worth exactly those derivations.
    let mut accumulator = if has_premise {
        DualPolynomial::zero()
    } else {
        DualPolynomial::one()
    };
    let mut height = 0usize;
    let mut cycle_truncated = false;
    for hit in &edges {
        let predecessor = fold_dual_recursive(
            store,
            &hit.node_id,
            depth + 1,
            ontology,
            memo,
            in_progress,
            expansions,
            caches,
        )?;
        cycle_truncated |= predecessor.cycle_truncated;
        let predecessor = predecessor.entry;
        height = height.max(predecessor.height + 1);
        let generator = mint_generator(store, &hit.edge_id, ontology)?;
        let defeating = generator
            .as_ref()
            .is_some_and(|generator| generator.polarity == Polarity::Negative);
        let edge_factor = generator
            .map(DualPolynomial::generator)
            .unwrap_or_else(DualPolynomial::one);
        let contribution = if defeating {
            // The self-loop case that used to open this chain is gone: a
            // defeating self-loop is a mutual pair with itself, so
            // `in_defeating_cycle` answers zero for the node before any edge
            // is walked and this loop never sees one.
            if !predecessor.polynomial.defeating_fragment().is_zero() {
                // The attacker is itself under attack, so it does not stand,
                // and an attack from a defeated attacker does not land. With C
                // undercutting B and B undercutting A, this is what leaves A
                // defended rather than `Out`.
                //
                // One level, matching the rule the structural pass applies and
                // the rule `DualReadout::into_grounded` reads back. Full Dung
                // reinstatement would ask whether C itself survives, which is a
                // fixpoint this bottom-up fold does not compute.
                DualPolynomial::zero()
            } else {
                // An attack carries only as far as its attacker is supported.
                // Multiplying by the predecessor's whole polynomial dragged its
                // *defeating* fragment along, so `xbar_CB * xbar_BA` landed in
                // A's defeating fragment and counted as a second attack on A.
                edge_factor.times(&predecessor.polynomial.positive_fragment())
            }
        } else {
            edge_factor.times(&predecessor.polynomial)
        };
        accumulator = accumulator.plus(&contribution);
        if accumulator.monomials().len() > MAX_DUAL_MONOMIALS {
            return Err(DualFoldError::TermBudgetExceeded {
                node_id: node_id.to_string(),
                monomials: accumulator.monomials().len(),
                limit: MAX_DUAL_MONOMIALS,
            });
        }
    }
    in_progress.remove(node_id);
    let entry = MemoEntry {
        polynomial: accumulator,
        height,
    };
    // Cacheable only when nothing beneath was cut. Otherwise this value is
    // true only under the ancestors currently on the stack, so leave the memo
    // alone and let a later caller recompute in its own context.
    if !cycle_truncated {
        memo.insert(node_id.to_string(), entry.clone());
    }
    Ok(FoldOutcome {
        entry,
        cycle_truncated,
    })
}

/// Resolve one edge into the generator it mints, or `None` when its class is
/// never valued.
fn mint_generator(
    store: &impl DerivationStore,
    edge_id: &str,
    ontology: &EdgeOntology,
) -> Result<Option<Generator>, DualFoldError> {
    let Some(edge_type) = store.edge_type(edge_id) else {
        return Err(DualFoldError::UntypedEdge {
            edge_id: edge_id.to_string(),
        });
    };
    let Some(spec) = ontology.resolve(&edge_type) else {
        return Err(DualFoldError::UnclassifiedEdge {
            edge_id: edge_id.to_string(),
            edge_type,
        });
    };
    Ok(Polarity::for_class(spec.class).map(|polarity| Generator {
        token: EdgeToken::new(edge_id),
        polarity,
        canonical_type: spec.canonical.clone(),
    }))
}

#[cfg(test)]
mod dual_refusal_tests {
    use super::*;
    use crate::ontology::{AlgebraicClass, EdgeOntology, EdgeTypeSpec};
    use crate::store::DerivationNeighbor;

    /// A chain of `depth` derivation nodes ending at "n0".
    struct Chain {
        depth: usize,
    }

    impl DerivationStore for Chain {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            let index: usize = node_id.strip_prefix('n')?.parse().ok()?;
            if index >= self.depth {
                Some(vec![ASSUMPTION_LABEL.to_string()])
            } else {
                Some(vec![DERIVATION_LABEL.to_string()])
            }
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if edge_type != DERIVATION_ASSUMES_EDGE {
                return Vec::new();
            }
            let Some(index) = node_id
                .strip_prefix('n')
                .and_then(|i| i.parse::<usize>().ok())
            else {
                return Vec::new();
            };
            if index >= self.depth {
                return Vec::new();
            }
            vec![DerivationNeighbor {
                node_id: format!("n{}", index + 1),
                edge_id: format!("e{index}"),
            }]
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some(DERIVATION_ASSUMES_EDGE.to_string())
        }
    }

    /// A node with `width` alternative single-edge derivations, each of which
    /// recurses into another such node. Monomials multiply per layer.
    struct Fanout {
        width: usize,
        layers: usize,
    }

    impl DerivationStore for Fanout {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            let layer: usize = node_id.split(':').next()?.parse().ok()?;
            if layer >= self.layers {
                Some(vec![ASSUMPTION_LABEL.to_string()])
            } else {
                Some(vec![DERIVATION_LABEL.to_string()])
            }
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if edge_type != DERIVATION_ASSUMES_EDGE {
                return Vec::new();
            }
            let Some(layer) = node_id
                .split(':')
                .next()
                .and_then(|l| l.parse::<usize>().ok())
            else {
                return Vec::new();
            };
            if layer >= self.layers {
                return Vec::new();
            }
            (0..self.width)
                .map(|branch| DerivationNeighbor {
                    node_id: format!("{}:{}:{branch}", layer + 1, node_id),
                    edge_id: format!("e:{node_id}:{branch}"),
                })
                .collect()
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some(DERIVATION_ASSUMES_EDGE.to_string())
        }
    }

    /// `target` reaches a deep shared suffix two ways: a short branch and a
    /// long one. The short branch is returned first, so it caches the suffix
    /// before the long branch asks for it.
    ///
    /// target -> s -> c0 -> ... -> c249 -> a        (deepest node at depth 252)
    /// target -> l0 -> ... -> l19 -> c0 -> ... -> a (deepest node at depth 271)
    ///
    /// The short path fits inside `MAX_DERIVATION_DEPTH`; the long one does
    /// not, so the fold must refuse whichever branch it walks first.
    struct SharedSuffix {
        long_branch: usize,
        suffix: usize,
    }

    impl DerivationStore for SharedSuffix {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            if node_id == "a" {
                Some(vec![ASSUMPTION_LABEL.to_string()])
            } else {
                Some(vec![DERIVATION_LABEL.to_string()])
            }
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if edge_type != DERIVATION_ASSUMES_EDGE {
                return Vec::new();
            }
            let step = |to: String| {
                vec![DerivationNeighbor {
                    edge_id: format!("e:{node_id}:{to}"),
                    node_id: to,
                }]
            };
            match node_id {
                // Short branch first: it caches the suffix.
                "target" => vec![
                    DerivationNeighbor {
                        node_id: "s".to_string(),
                        edge_id: "e:target:s".to_string(),
                    },
                    DerivationNeighbor {
                        node_id: "l0".to_string(),
                        edge_id: "e:target:l0".to_string(),
                    },
                ],
                "s" => step("c0".to_string()),
                "a" => Vec::new(),
                _ => {
                    if let Some(index) = node_id
                        .strip_prefix('l')
                        .and_then(|i| i.parse::<usize>().ok())
                    {
                        return if index + 1 < self.long_branch {
                            step(format!("l{}", index + 1))
                        } else {
                            step("c0".to_string())
                        };
                    }
                    if let Some(index) = node_id
                        .strip_prefix('c')
                        .and_then(|i| i.parse::<usize>().ok())
                    {
                        return if index + 1 < self.suffix {
                            step(format!("c{}", index + 1))
                        } else {
                            step("a".to_string())
                        };
                    }
                    Vec::new()
                }
            }
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some(DERIVATION_ASSUMES_EDGE.to_string())
        }
    }

    /// Codex's cycle fixture: `P -> A -> B`, `B -> A`, and both `A -> T` and
    /// `B -> T`. `A` and `B` sit in a cycle with an acyclic exit, and both are
    /// reachable straight from the target, which is what exposes a cached
    /// cycle-truncated value. `reversed` flips the order `T`'s predecessors
    /// come back in; the fold must answer the same either way.
    struct CycleWithExit {
        reversed: bool,
    }

    impl DerivationStore for CycleWithExit {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            if node_id == "P" {
                Some(vec![ASSUMPTION_LABEL.to_string()])
            } else {
                Some(vec![DERIVATION_LABEL.to_string()])
            }
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if edge_type != DERIVATION_ASSUMES_EDGE {
                return Vec::new();
            }
            let edge = |from: &str, to: &str| DerivationNeighbor {
                node_id: from.to_string(),
                edge_id: format!("e:{from}->{to}"),
            };
            match node_id {
                "T" => {
                    let mut preds = vec![edge("A", "T"), edge("B", "T")];
                    if self.reversed {
                        preds.reverse();
                    }
                    preds
                }
                // P -> A and B -> A
                "A" => vec![edge("P", "A"), edge("B", "A")],
                // A -> B
                "B" => vec![edge("A", "B")],
                _ => Vec::new(),
            }
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some(DERIVATION_ASSUMES_EDGE.to_string())
        }
    }

    /// A complete cyclic component with no assumption anywhere in it. Every
    /// node reaches every other, so no polynomial is ever nonzero and no
    /// derivation exists at all, yet the traversal still has to walk every
    /// simple path because a cycle disables memoization.
    struct DenseCycle {
        nodes: usize,
    }

    impl DerivationStore for DenseCycle {
        fn node_labels(&self, _node_id: &str) -> Option<Vec<String>> {
            Some(vec![DERIVATION_LABEL.to_string()])
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if edge_type != DERIVATION_ASSUMES_EDGE {
                return Vec::new();
            }
            let Some(index) = node_id
                .strip_prefix('n')
                .and_then(|i| i.parse::<usize>().ok())
            else {
                return Vec::new();
            };
            (0..self.nodes)
                .filter(|other| *other != index)
                .map(|other| DerivationNeighbor {
                    node_id: format!("n{other}"),
                    edge_id: format!("e:{other}->{index}"),
                })
                .collect()
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some(DERIVATION_ASSUMES_EDGE.to_string())
        }
    }

    /// A single ring of `len` derivations: `n{i}` is derived from
    /// `n{(i+1) % len}`. Folding `n0` walks the ring and meets `n0` again at
    /// depth `len`.
    struct Ring {
        len: usize,
    }

    impl DerivationStore for Ring {
        fn node_labels(&self, _node_id: &str) -> Option<Vec<String>> {
            Some(vec![DERIVATION_LABEL.to_string()])
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if edge_type != DERIVATION_ASSUMES_EDGE {
                return Vec::new();
            }
            let Some(index) = node_id
                .strip_prefix('n')
                .and_then(|i| i.parse::<usize>().ok())
            else {
                return Vec::new();
            };
            let next = (index + 1) % self.len;
            vec![DerivationNeighbor {
                node_id: format!("n{next}"),
                edge_id: format!("e:{next}->{index}"),
            }]
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some(DERIVATION_ASSUMES_EDGE.to_string())
        }
    }

    /// An assumption `a` attacked by another assumption `b`. The attack is a
    /// real defeating edge, so the ontology must be one that carries the
    /// defeating rows.
    struct AttackedAssumption;

    impl DerivationStore for AttackedAssumption {
        fn node_labels(&self, _node_id: &str) -> Option<Vec<String>> {
            Some(vec![ASSUMPTION_LABEL.to_string()])
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if node_id != "a" || edge_type != "Undercuts" {
                return Vec::new();
            }
            vec![DerivationNeighbor {
                node_id: "b".to_string(),
                edge_id: "e:b->a".to_string(),
            }]
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some("Undercuts".to_string())
        }
    }

    /// An assumption that attacks itself. Grounded semantics forbids calling a
    /// self-attacking argument accepted.
    struct SelfAttackingAssumption;

    impl DerivationStore for SelfAttackingAssumption {
        fn node_labels(&self, _node_id: &str) -> Option<Vec<String>> {
            Some(vec![ASSUMPTION_LABEL.to_string()])
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if node_id != "a" || edge_type != "Undercuts" {
                return Vec::new();
            }
            vec![DerivationNeighbor {
                node_id: "a".to_string(),
                edge_id: "e:a->a".to_string(),
            }]
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some("Undercuts".to_string())
        }
    }

    /// `C` undercuts `B`, `B` undercuts `A`. `A` rests on premise `P` and `B`
    /// rests on premise `Q`, so `B` has real positive support of its own while
    /// being attacked. Grounded semantics puts `B` out and leaves `A`
    /// defended.
    ///
    /// `B` needs `Q`. Without it `B` has no positive fragment and `A` comes out
    /// defended for the wrong reason, which is exactly how the round 16
    /// fixture passed while the defect was still live.
    struct DefendedTarget;

    impl DerivationStore for DefendedTarget {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            if node_id == "P" || node_id == "Q" || node_id == "C" {
                Some(vec![ASSUMPTION_LABEL.to_string()])
            } else {
                Some(vec![DERIVATION_LABEL.to_string()])
            }
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            match (node_id, edge_type) {
                ("A", DERIVATION_ASSUMES_EDGE) => vec![DerivationNeighbor {
                    node_id: "P".to_string(),
                    edge_id: "e:P->A".to_string(),
                }],
                ("A", "Undercuts") => vec![DerivationNeighbor {
                    node_id: "B".to_string(),
                    edge_id: "e:B->A".to_string(),
                }],
                ("B", DERIVATION_ASSUMES_EDGE) => vec![DerivationNeighbor {
                    node_id: "Q".to_string(),
                    edge_id: "e:Q->B".to_string(),
                }],
                ("B", "Undercuts") => vec![DerivationNeighbor {
                    node_id: "C".to_string(),
                    edge_id: "e:C->B".to_string(),
                }],
                _ => Vec::new(),
            }
        }
        fn edge_type(&self, edge_id: &str) -> Option<String> {
            if edge_id == "e:P->A" || edge_id == "e:Q->B" {
                Some(DERIVATION_ASSUMES_EDGE.to_string())
            } else {
                Some("Undercuts".to_string())
            }
        }
    }

    /// Same shape as `DefendedTarget`, but C's attack on B carries annotation
    /// `0.0`. A zero-weight attack asserts nothing, so B is not defeated and
    /// B's real attack on A must survive.
    struct ZeroWeightDefenderTarget;

    impl DerivationStore for ZeroWeightDefenderTarget {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            if node_id == "P" || node_id == "Q" || node_id == "C" {
                Some(vec![ASSUMPTION_LABEL.to_string()])
            } else {
                Some(vec![DERIVATION_LABEL.to_string()])
            }
        }
        fn edge_annotation(&self, edge_id: &str, _property: &str) -> Option<f64> {
            if edge_id == "e:C->B" {
                Some(0.0)
            } else {
                Some(1.0)
            }
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            match (node_id, edge_type) {
                ("A", DERIVATION_ASSUMES_EDGE) => vec![DerivationNeighbor {
                    node_id: "P".to_string(),
                    edge_id: "e:P->A".to_string(),
                }],
                ("A", "Undercuts") => vec![DerivationNeighbor {
                    node_id: "B".to_string(),
                    edge_id: "e:B->A".to_string(),
                }],
                ("B", DERIVATION_ASSUMES_EDGE) => vec![DerivationNeighbor {
                    node_id: "Q".to_string(),
                    edge_id: "e:Q->B".to_string(),
                }],
                ("B", "Undercuts") => vec![DerivationNeighbor {
                    node_id: "C".to_string(),
                    edge_id: "e:C->B".to_string(),
                }],
                _ => Vec::new(),
            }
        }
        fn edge_type(&self, edge_id: &str) -> Option<String> {
            if edge_id == "e:P->A" || edge_id == "e:Q->B" {
                Some(DERIVATION_ASSUMES_EDGE.to_string())
            } else {
                Some("Undercuts".to_string())
            }
        }
    }

    /// A premise-free derivation with an incoming attack. The scalar fold
    /// answers `one()` for a premise-free derivation, so the dual fold must
    /// keep its unit rather than losing it because an attacker exists.
    struct AttackedPremiseFreeDerivation;

    impl DerivationStore for AttackedPremiseFreeDerivation {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            if node_id == "E" {
                Some(vec![ASSUMPTION_LABEL.to_string()])
            } else {
                Some(vec![DERIVATION_LABEL.to_string()])
            }
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if node_id == "D" && edge_type == "Undercuts" {
                vec![DerivationNeighbor {
                    node_id: "E".to_string(),
                    edge_id: "e:E->D".to_string(),
                }]
            } else {
                Vec::new()
            }
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some("Undercuts".to_string())
        }
    }

    /// `B` has only a positive self-cycle, so it has no derivation at all, and
    /// `B` undercuts assumption `A`. An attacker with no grounded premise must
    /// not defeat its target.
    struct UnsupportedCyclicAttacker;

    impl DerivationStore for UnsupportedCyclicAttacker {
        fn node_labels(&self, node_id: &str) -> Option<Vec<String>> {
            if node_id == "A" {
                Some(vec![ASSUMPTION_LABEL.to_string()])
            } else {
                Some(vec![DERIVATION_LABEL.to_string()])
            }
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            match (node_id, edge_type) {
                ("A", "Undercuts") => vec![DerivationNeighbor {
                    node_id: "B".to_string(),
                    edge_id: "e:B->A".to_string(),
                }],
                ("B", DERIVATION_ASSUMES_EDGE) => vec![DerivationNeighbor {
                    node_id: "B".to_string(),
                    edge_id: "e:B->B".to_string(),
                }],
                _ => Vec::new(),
            }
        }
        fn edge_type(&self, edge_id: &str) -> Option<String> {
            if edge_id == "e:B->A" {
                Some("Undercuts".to_string())
            } else {
                Some(DERIVATION_ASSUMES_EDGE.to_string())
            }
        }
    }

    /// `len` assumptions in one directed defeating ring: n{i} attacks
    /// n{(i+1) % len}, so the attacker of n{i} is n{(i + len - 1) % len}.
    /// A layered acyclic defeating DAG: every node in layer k is attacked by
    /// every node in layer k+1. No cycles, and every node shares predecessors
    /// with its siblings, which is the shape that made the reachability pass
    /// re-walk the same subgraph once per node. Counts every neighbour probe.
    struct DenseDefeatingDag {
        width: usize,
        layers: usize,
        probes: std::cell::Cell<usize>,
    }

    impl DerivationStore for DenseDefeatingDag {
        fn node_labels(&self, _node_id: &str) -> Option<Vec<String>> {
            Some(vec![ASSUMPTION_LABEL.to_string()])
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if edge_type != "Undercuts" {
                return Vec::new();
            }
            self.probes.set(self.probes.get() + 1);
            let Some((layer, _)) = node_id
                .strip_prefix('l')
                .and_then(|rest| rest.split_once('n'))
                .and_then(|(l, i)| Some((l.parse::<usize>().ok()?, i)))
            else {
                return Vec::new();
            };
            if layer + 1 >= self.layers {
                return Vec::new();
            }
            (0..self.width)
                .map(|j| DerivationNeighbor {
                    node_id: format!("l{}n{j}", layer + 1),
                    edge_id: format!("e:l{}n{j}->{node_id}", layer + 1),
                })
                .collect()
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some("Undercuts".to_string())
        }
    }

    struct DefeatingRing {
        len: usize,
    }

    impl DerivationStore for DefeatingRing {
        fn node_labels(&self, _node_id: &str) -> Option<Vec<String>> {
            Some(vec![ASSUMPTION_LABEL.to_string()])
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            if edge_type != "Undercuts" {
                return Vec::new();
            }
            let Some(index) = node_id
                .strip_prefix('n')
                .and_then(|i| i.parse::<usize>().ok())
            else {
                return Vec::new();
            };
            let attacker = (index + self.len - 1) % self.len;
            vec![DerivationNeighbor {
                node_id: format!("n{attacker}"),
                edge_id: format!("e:{attacker}->{index}"),
            }]
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some("Undercuts".to_string())
        }
    }

    struct MutualAttack;

    impl DerivationStore for MutualAttack {
        fn node_labels(&self, _node_id: &str) -> Option<Vec<String>> {
            Some(vec![ASSUMPTION_LABEL.to_string()])
        }
        fn edge_annotation(&self, _edge_id: &str, _property: &str) -> Option<f64> {
            Some(1.0)
        }
        fn neighbors_in(&self, node_id: &str, edge_type: &str) -> Vec<DerivationNeighbor> {
            match (node_id, edge_type) {
                ("A", "Undercuts") => vec![DerivationNeighbor {
                    node_id: "B".to_string(),
                    edge_id: "e:B->A".to_string(),
                }],
                ("B", "Undercuts") => vec![DerivationNeighbor {
                    node_id: "A".to_string(),
                    edge_id: "e:A->B".to_string(),
                }],
                _ => Vec::new(),
            }
        }
        fn edge_type(&self, _edge_id: &str) -> Option<String> {
            Some("Undercuts".to_string())
        }
    }

    fn ontology() -> EdgeOntology {
        EdgeOntology::from_specs(vec![EdgeTypeSpec::new(
            DERIVATION_ASSUMES_EDGE,
            AlgebraicClass::MonotonePositive,
            Vec::<String>::new(),
            None,
        )])
        .expect("test ontology")
    }

    #[test]
    fn a_shallow_chain_folds_without_refusing() {
        let poly = fold_dual(&Chain { depth: 8 }, "n0", &ontology()).expect("within bounds");
        assert_eq!(poly.monomials().len(), 1);
        assert_eq!(poly.monomials()[0].generators.len(), 8);
    }

    #[test]
    fn past_the_depth_bound_the_fold_refuses_instead_of_answering_zero() {
        // The scalar fold applies its depth bound only to non-absorptive
        // semirings, so Boolean here still reaches the assumption and answers
        // true. A zero polynomial would read as "no derivation" and contradict
        // it, so the dual fold refuses rather than returning a false zero.
        let deep = Chain {
            depth: MAX_DERIVATION_DEPTH + 8,
        };
        assert!(
            fold::<crate::semiring::BooleanSemiring>(&deep, "n0"),
            "the absorptive scalar fold still finds the derivation"
        );
        let err = fold_dual(&deep, "n0", &ontology()).unwrap_err();
        assert!(
            matches!(err, DualFoldError::DepthExceeded { limit, .. } if limit == MAX_DERIVATION_DEPTH),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn a_memoized_suffix_does_not_smuggle_a_long_branch_past_the_depth_bound() {
        // The memo is keyed by node alone. Walking the short branch first
        // caches the shared suffix, and the long branch then reused that cache
        // even though its complete path runs past MAX_DERIVATION_DEPTH, so
        // whether an over-depth graph was accepted depended on the order the
        // store happened to return edges in. The cached height makes a hit
        // answer the same question a fresh traversal would.
        let store = SharedSuffix {
            long_branch: 20,
            suffix: 250,
        };

        // Sanity: the short branch alone is inside the bound.
        assert!(
            2 + store.suffix <= MAX_DERIVATION_DEPTH,
            "the short path must fit, or the fixture proves nothing"
        );
        // And the long branch alone is past it.
        assert!(
            1 + store.long_branch + store.suffix > MAX_DERIVATION_DEPTH,
            "the long path must not fit, or the fixture proves nothing"
        );

        let err = fold_dual(&store, "target", &ontology())
            .expect_err("the long branch runs past the depth bound");
        assert!(
            matches!(err, DualFoldError::DepthExceeded { limit, .. } if limit == MAX_DERIVATION_DEPTH),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn a_cycle_truncated_subtree_is_not_cached_as_if_it_were_settled() {
        // A cycle is cut by answering zero for a node already on the stack.
        // That zero holds only while that ancestor is being expanded, so
        // caching a value computed underneath it made the fold depend on
        // neighbor order: expanding A first cached B as zero and lost the
        // derivation P -> A -> B -> T, while expanding B first kept it.
        let forward = fold_dual(&CycleWithExit { reversed: false }, "T", &ontology())
            .expect("acyclic exit is within bounds");
        let reversed = fold_dual(&CycleWithExit { reversed: true }, "T", &ontology())
            .expect("acyclic exit is within bounds");

        assert_eq!(
            forward, reversed,
            "the polynomial must not depend on the order predecessors come back in"
        );
        // Both simple derivations survive: P -> A -> T and P -> A -> B -> T.
        assert_eq!(
            forward.monomials().len(),
            2,
            "expected both simple derivations, got {forward:?}"
        );
    }

    #[test]
    fn a_dense_cycle_refuses_on_the_expansion_budget_not_on_a_result_bound() {
        // Both other bounds watch the result. Here every polynomial is zero,
        // because no assumption is reachable, so the term budget cannot fire;
        // and ten nodes sit far below MAX_DERIVATION_DEPTH, so the depth bound
        // cannot either. Without a work bound this enumerates all 986410
        // simple paths.
        let store = DenseCycle { nodes: 10 };
        let err = fold_dual(&store, "n0", &ontology()).expect_err("must refuse, not run away");
        assert!(
            matches!(
                err,
                DualFoldError::ExpansionBudgetExceeded { limit, .. }
                    if limit == MAX_DUAL_EXPANSIONS
            ),
            "expected the work bound to fire, got {err:?}"
        );
    }

    #[test]
    fn an_attacked_assumption_still_mints_its_attacker() {
        // The assumption branch used to return the unit before edge collection
        // ran, so a defeating edge into an assumption never minted a negative
        // generator and a Boolean readout called the attacked assumption In.
        let poly = fold_dual(&AttackedAssumption, "a", derivation_dag_ontology())
            .expect("two assumptions and one edge is well inside every bound");

        assert!(
            !poly.positive_fragment().is_zero(),
            "the assumption's own unit must survive the attack"
        );
        assert!(
            !poly.defeating_fragment().is_zero(),
            "the attack must reach the against side, got {poly:?}"
        );
    }

    #[test]
    fn a_self_attacking_assumption_does_not_read_as_accepted() {
        // The cycle guard answers zero for the node already on the stack, and
        // multiplying the defeating edge by that zero left `1 + xbar * 0 = 1`,
        // so every readout called the self-attacking assumption In.
        //
        // It is not `Out` either. A defeating self-loop satisfies the same
        // reverse-edge test `contradiction_cycles` applies in the structural
        // pass, so the pass calls it undecided and this fold answers zero to
        // agree.
        let poly = fold_dual(&SelfAttackingAssumption, "a", derivation_dag_ontology())
            .expect("one node and one edge is inside every bound");

        assert!(
            poly.is_zero(),
            "a node standing in its own way is undecided, not accepted and not \
             defeated, got {poly:?}"
        );
    }

    #[test]
    fn a_defeated_attacker_leaves_its_target_defended() {
        // C undercuts B and B undercuts A, so grounded semantics puts B out and
        // leaves A defended.
        //
        // The round 16 fix stopped B's own defeating fragment leaking into A's
        // monomials, and the round 16 test checked only that. It did not check
        // that A ends up defended, and B in that fixture had no premise, so A
        // came out right for the wrong reason while the real defect was still
        // live. Assert the standing, not the omission.
        let poly = fold_dual(&DefendedTarget, "A", derivation_dag_ontology())
            .expect("five nodes is inside every bound");

        assert!(
            !poly.positive_fragment().is_zero(),
            "A still rests on its own premise"
        );
        assert!(
            poly.defeating_fragment().is_zero(),
            "a defeated attacker must not leave its target attacked, got {poly:?}"
        );

        // And B itself is attacked, so the fixture really does exercise the
        // case rather than passing because B had nothing to say.
        let b = fold_dual(&DefendedTarget, "B", derivation_dag_ontology())
            .expect("five nodes is inside every bound");
        assert!(
            !b.positive_fragment().is_zero(),
            "B must have real support of its own, or the fixture proves nothing"
        );
        assert!(
            !b.defeating_fragment().is_zero(),
            "B must actually be under attack"
        );
    }

    #[test]
    fn a_zero_weight_defender_does_not_cancel_a_live_attack() {
        // Suppressing a defeated attacker keyed on the attacker's defeating
        // fragment being structurally non-empty. A zero-weight attack asserts
        // nothing, yet it still minted a generator, so a weightless C -> B was
        // enough to make B look defeated and silently cancel B's real attack
        // on A.
        let poly = fold_dual(&ZeroWeightDefenderTarget, "A", derivation_dag_ontology())
            .expect("five nodes is inside every bound");

        assert!(
            !poly.defeating_fragment().is_zero(),
            "a weightless defender must not cancel a live attack, got {poly:?}"
        );

        // The same fixture with a weighted defender does defend, so the two
        // cases really are distinguished by the annotation.
        let defended = fold_dual(&DefendedTarget, "A", derivation_dag_ontology())
            .expect("five nodes is inside every bound");
        assert!(
            defended.defeating_fragment().is_zero(),
            "a weighted defender still defends"
        );
    }

    #[test]
    fn an_attacked_premise_free_derivation_keeps_its_unit() {
        // Collecting the attack made `edges` nonempty, and the accumulator was
        // seeded from that rather than from whether any premise reached the
        // node, so a premise-free derivation lost its unit the moment an
        // attacker appeared. The scalar fold answers one() for the same node.
        let poly = fold_dual(
            &AttackedPremiseFreeDerivation,
            "D",
            derivation_dag_ontology(),
        )
        .expect("two nodes is inside every bound");

        assert!(
            !poly.positive_fragment().is_zero(),
            "a premise-free derivation keeps its unit under attack, got {poly:?}"
        );
        assert!(
            !poly.defeating_fragment().is_zero(),
            "and the attack still lands"
        );
    }

    #[test]
    fn an_unsupported_cyclic_attacker_does_not_defeat_its_target() {
        // B has only a positive self-cycle, so it has no derivation. The
        // self-attack rule must not revive its edge.
        let poly = fold_dual(&UnsupportedCyclicAttacker, "A", derivation_dag_ontology())
            .expect("two nodes is inside every bound");

        assert!(
            !poly.positive_fragment().is_zero(),
            "the assumption still stands"
        );
        assert!(
            poly.defeating_fragment().is_zero(),
            "an attacker with no derivation must not defeat its target, got {poly:?}"
        );
    }

    #[test]
    fn a_defeating_ring_past_the_depth_bound_refuses_like_a_positive_one() {
        // The reachability prepass answers before the recursive fold runs, so
        // it can settle a ring the fold itself would refuse to walk. A positive
        // ring of 257 raises DepthExceeded; a defeating ring of the same width
        // must not quietly answer instead, or the bound means one thing on a
        // support chain and another on an attack chain.
        let err = fold_dual(
            &DefeatingRing { len: 257 },
            "n0",
            &derivation_dag_ontology(),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                DualFoldError::DepthExceeded { limit, .. } if limit == MAX_DERIVATION_DEPTH
            ),
            "expected a depth refusal, got {err:?}"
        );
    }

    #[test]
    fn the_cycle_search_is_answered_once_per_node() {
        // Every node the recursion reaches asks whether it sits on a defeating
        // cycle. On a dense acyclic defeating DAG the answer is no, but finding
        // that out walks the whole subgraph below the node, and siblings share
        // almost all of it. Uncached, the probes grow with the product of the
        // node count and the subgraph size.
        let store = DenseDefeatingDag {
            width: 6,
            layers: 6,
            probes: std::cell::Cell::new(0),
        };
        fold_dual(&store, "l0n0", &derivation_dag_ontology()).expect("inside every bound");

        // 6 layers of 6 is 31 distinct nodes reachable from the anchor. One
        // cycle answer and one edge collection each is the floor; the bound
        // here is comfortably above that and far below the uncached cost,
        // which exceeds 400 probes for this shape.
        let probes = store.probes.get();
        assert!(
            probes < 200,
            "cycle membership must be answered once per node, took {probes} probes"
        );
    }

    #[test]
    fn a_longer_defeating_ring_reads_undecided_like_a_mutual_pair() {
        // The reverse-edge test that mirrored `contradiction_cycles` saw only
        // self-loops and mutual pairs, so a longer ring fell through to the
        // one-level rule and answered by parity: three nodes suppressed every
        // attack and read In, four left one attack standing and read Out.
        // Neither is defensible, and a standing that flips on whether a ring
        // is odd is the tell that cycles were not being treated as cycles.
        //
        // Grounded semantics puts every node of an unsupported attack cycle in
        // the undecided set, which is the zero polynomial here.
        for len in [3usize, 4, 5] {
            let poly = fold_dual(&DefeatingRing { len }, "n0", &derivation_dag_ontology())
                .expect("a small ring is inside every bound");
            assert!(
                poly.is_zero(),
                "a node on a {len}-node defeating ring must read undecided, got {poly:?}"
            );
        }
    }

    #[test]
    fn mutual_defeat_reads_undecided_like_the_structural_pass() {
        // A and B attack each other and nothing else. `contradiction_cycles`
        // pairs each attack with its reverse and marks both ends, and
        // `structural_epistemic_pass` checks that cycle membership *before* it
        // checks attack degree, so both nodes are undecided rather than out.
        //
        // This fold used to disagree. The cycle guard cut whichever return leg
        // the traversal reached second, so the other attacker's defeating
        // fragment looked empty, its attack was allowed to land, and both
        // anchors read `Out` -- a standing decided by traversal order.
        for anchor in ["A", "B"] {
            let poly = fold_dual(&MutualAttack, anchor, derivation_dag_ontology())
                .expect("two nodes is inside every bound");
            assert!(
                poly.is_zero(),
                "{anchor} is in a mutual defeating pair and must read undecided, \
                 got {poly:?}"
            );
        }
    }
    #[test]
    fn a_cycle_longer_than_the_depth_bound_refuses_rather_than_truncating() {
        // MAX_DERIVATION_DEPTH is 256, so a ring of 257 nodes is the exact
        // width where the two guards disagree: every intermediate node sits at
        // depth 256 or less, and the walk meets the requested node again at
        // depth 257. Checking `in_progress` first swallowed that as a
        // cycle-truncated zero, so the traversal ran past a bound the contract
        // calls a refusal. Any wider ring refuses at an intermediate node under
        // either ordering and would not show the difference.
        let err = fold_dual(&Ring { len: 257 }, "n0", &ontology()).unwrap_err();
        assert!(
            matches!(
                err,
                DualFoldError::DepthExceeded { limit, .. } if limit == MAX_DERIVATION_DEPTH
            ),
            "expected a depth refusal, got {err:?}"
        );
    }

    #[test]
    fn runaway_path_expansion_refuses_instead_of_exhausting_memory() {
        // 4 branches per layer over 8 layers is 65536 complete paths, well past
        // the bound, and it must refuse rather than materialize them.
        let err = fold_dual(
            &Fanout {
                width: 4,
                layers: 8,
            },
            "0:root",
            &ontology(),
        )
        .unwrap_err();
        assert!(
            matches!(err, DualFoldError::TermBudgetExceeded { limit, monomials, .. }
                if limit == MAX_DUAL_MONOMIALS && monomials > MAX_DUAL_MONOMIALS),
            "unexpected error: {err:?}"
        );
    }
}
