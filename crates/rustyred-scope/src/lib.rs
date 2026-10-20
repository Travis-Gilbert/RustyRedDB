//! The capability scope: the subset of affordances an agent is primed for.
//!
//! # Why this is its own crate
//!
//! ADDENDUM-2 A9/D10 binds the composer's capability dial to *this* type -
//! "re-exported, not mirrored" - so a dial entry and a run's affordance gate
//! cannot drift into two vocabularies. The type used to live in
//! `rustyred-thg-affordances`, which reaches the graph store, the plugin host,
//! and the harness kernel. `theorem-surface-contracts` is compiled into a
//! `wasm32-unknown-unknown` leaf and is dependency-light on purpose, so it
//! cannot take that edge; mirroring the type instead is exactly what the
//! addendum forbids.
//!
//! So the charter moves down here, where it only ever needed `serde` and four
//! string lists, and `rustyred-thg-affordances` re-exports it. Every existing
//! `rustyred_thg_affordances::CapabilityScope` path still resolves, and every
//! existing `scope.admits(&affordance)` call still compiles, because [`admits`]
//! is an inherent method generic over [`Admissible`] rather than a trait method
//! callers would have to import.
//!
//! [`admits`]: CapabilityScope::admits

use serde::{Deserialize, Serialize};

pub mod browser;

/// What a scope needs to know about a thing to admit it.
///
/// Four accessors rather than the whole affordance record, because those are
/// the four dimensions the intersection reads and every other field on an
/// affordance is graph-shaped.
pub trait Admissible {
    /// The affordance's stable identity.
    fn scope_affordance_id(&self) -> &str;
    /// The server it came from; empty for a built-in engine.
    fn scope_server_id(&self) -> &str;
    /// Its family - `browse`, `fs`, `graph`, `code`, and so on.
    fn scope_family(&self) -> &str;
    /// Free tags, matched as a set intersection.
    fn scope_tags(&self) -> &[String];
}

impl<T: Admissible + ?Sized> Admissible for &T {
    fn scope_affordance_id(&self) -> &str {
        T::scope_affordance_id(*self)
    }
    fn scope_server_id(&self) -> &str {
        T::scope_server_id(*self)
    }
    fn scope_family(&self) -> &str {
        T::scope_family(*self)
    }
    fn scope_tags(&self) -> &[String] {
        T::scope_tags(*self)
    }
}

/// The capability-scope plane of the AgentBinding: the subset of affordances an
/// agent is primed for. Selection filters to this scope, then the learned prior
/// ranks within it. An unrestricted scope (all lists empty) admits everything;
/// any populated dimension imposes an intersection constraint, so a scope can
/// be `allow_servers=["github"]` AND `allow_tags=["write"]` for github write
/// tools only.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
pub struct CapabilityScope {
    pub agent_id: String,
    #[serde(default)]
    pub allow_affordance_ids: Vec<String>,
    #[serde(default)]
    pub allow_servers: Vec<String>,
    #[serde(default)]
    pub allow_families: Vec<String>,
    #[serde(default)]
    pub allow_tags: Vec<String>,
}

impl CapabilityScope {
    #[must_use]
    pub fn unrestricted(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            ..Default::default()
        }
    }

    #[must_use]
    pub fn is_unrestricted(&self) -> bool {
        self.allow_affordance_ids.is_empty()
            && self.allow_servers.is_empty()
            && self.allow_families.is_empty()
            && self.allow_tags.is_empty()
    }

    /// Admits an affordance if it satisfies every populated allowlist dimension.
    pub fn admits<A: Admissible + ?Sized>(&self, affordance: &A) -> bool {
        if self.is_unrestricted() {
            return true;
        }
        if !self.allow_affordance_ids.is_empty()
            && !self
                .allow_affordance_ids
                .iter()
                .any(|id| id == affordance.scope_affordance_id())
        {
            return false;
        }
        if !self.allow_servers.is_empty()
            && !self
                .allow_servers
                .iter()
                .any(|server| server == affordance.scope_server_id())
        {
            return false;
        }
        if !self.allow_families.is_empty()
            && !self
                .allow_families
                .iter()
                .any(|family| family == affordance.scope_family())
        {
            return false;
        }
        if !self.allow_tags.is_empty()
            && !self
                .allow_tags
                .iter()
                .any(|tag| affordance.scope_tags().iter().any(|owned| owned == tag))
        {
            return false;
        }
        true
    }

    /// The union of two scopes, as the composer's dial builds a run's reach.
    ///
    /// Each dimension is concatenated, deduplicated and sorted, so a turn's
    /// scope is a value rather than a history of the order the person clicked
    /// in. `agent_id` is taken from `self`, because a union of two agents'
    /// scopes is not a thing the dial can produce.
    #[must_use]
    pub fn union(mut self, other: &Self) -> Self {
        for (own, other) in [
            (&mut self.allow_affordance_ids, &other.allow_affordance_ids),
            (&mut self.allow_servers, &other.allow_servers),
            (&mut self.allow_families, &other.allow_families),
            (&mut self.allow_tags, &other.allow_tags),
        ] {
            own.extend(other.iter().cloned());
            own.sort();
            own.dedup();
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tool {
        id: String,
        server: String,
        family: String,
        tags: Vec<String>,
    }

    impl Admissible for Tool {
        fn scope_affordance_id(&self) -> &str {
            &self.id
        }
        fn scope_server_id(&self) -> &str {
            &self.server
        }
        fn scope_family(&self) -> &str {
            &self.family
        }
        fn scope_tags(&self) -> &[String] {
            &self.tags
        }
    }

    fn tool(family: &str) -> Tool {
        Tool {
            id: "recall".to_owned(),
            server: String::new(),
            family: family.to_owned(),
            tags: vec!["read".to_owned()],
        }
    }

    #[test]
    fn an_empty_scope_admits_everything() {
        assert!(CapabilityScope::unrestricted("agent").admits(&tool("graph")));
    }

    #[test]
    fn a_populated_dimension_is_an_intersection_not_a_union() {
        let scope = CapabilityScope {
            allow_families: vec!["graph".to_owned()],
            allow_tags: vec!["write".to_owned()],
            ..CapabilityScope::unrestricted("agent")
        };
        // Family matches, tag does not: both must hold.
        assert!(!scope.admits(&tool("graph")));
    }

    #[test]
    fn the_union_is_sorted_and_order_independent() {
        let left = CapabilityScope {
            allow_families: vec!["recall".to_owned(), "graph".to_owned()],
            ..CapabilityScope::unrestricted("agent")
        };
        let right = CapabilityScope {
            allow_families: vec!["code".to_owned(), "graph".to_owned()],
            ..CapabilityScope::unrestricted("agent")
        };
        assert_eq!(
            left.clone().union(&right).allow_families,
            vec!["code", "graph", "recall"]
        );
        assert_eq!(
            right.union(&left).allow_families,
            vec!["code", "graph", "recall"]
        );
    }
}
