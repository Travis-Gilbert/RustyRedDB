//! Opaque capability resolution for host-owned browser storage.
//!
//! This module deliberately knows nothing about Servo, RustyRed, origins, or
//! partitions.  Those concrete types live above this dependency-light crate.
//! Its one job is to preserve the security property that a caller-held token
//! can resolve only to authority the host registered for that exact token.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

/// A host-side table from opaque caller-held tokens to private authority.
///
/// `A` is intentionally generic.  The desktop broker stores the frozen
/// `StorageGrant` contract here, while this leaf crate remains independent of
/// the surface-contract crate which already depends on it.
#[derive(Clone, Debug)]
pub struct OpaqueGrantTable<A> {
    entries: BTreeMap<String, A>,
}

impl<A> Default for OpaqueGrantTable<A> {
    fn default() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }
}

impl<A> OpaqueGrantTable<A> {
    /// Register a freshly minted opaque token.
    ///
    /// Empty and duplicate tokens are refused.  In particular, registering a
    /// duplicate must never replace the authority behind an already issued
    /// capability.
    pub fn register(
        &mut self,
        token: impl Into<String>,
        authority: A,
    ) -> Result<(), OpaqueGrantTableError> {
        let token = token.into();
        if token.is_empty() {
            return Err(OpaqueGrantTableError::EmptyToken);
        }
        if self.entries.contains_key(&token) {
            return Err(OpaqueGrantTableError::DuplicateToken);
        }
        self.entries.insert(token, authority);
        Ok(())
    }

    /// Resolve exactly one token.  A forged value is indistinguishable from a
    /// revoked value and returns no authority.
    #[must_use]
    pub fn resolve(&self, token: &str) -> Option<&A> {
        self.entries.get(token)
    }

    /// Resolve only when the host's private authority satisfies `predicate`.
    ///
    /// This is the fail-closed crossing check used when a host adapter already
    /// has an expected partition, origin, database, or residency.  A mismatch
    /// returns the same `None` as an unknown token rather than leaking which
    /// private dimension differed.
    #[must_use]
    pub fn resolve_matching(&self, token: &str, predicate: impl FnOnce(&A) -> bool) -> Option<&A> {
        self.resolve(token).filter(|authority| predicate(authority))
    }

    /// Revoke one capability, returning its private authority to the host.
    pub fn revoke(&mut self, token: &str) -> Option<A> {
        self.entries.remove(token)
    }

    /// Revoke every capability selected by a host-side authority predicate.
    ///
    /// The caller never supplies this predicate; it is for lifecycle events
    /// such as destroying an ephemeral browser partition.
    pub fn revoke_where(&mut self, predicate: impl Fn(&A) -> bool) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, authority| !predicate(authority));
        before - self.entries.len()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpaqueGrantTableError {
    EmptyToken,
    DuplicateToken,
}

impl fmt::Display for OpaqueGrantTableError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::EmptyToken => "opaque grant token cannot be empty",
            Self::DuplicateToken => "opaque grant token is already registered",
        })
    }
}

impl Error for OpaqueGrantTableError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct BrowserAuthority {
        partition: &'static str,
        origin: &'static str,
        database: &'static str,
        ephemeral: bool,
    }

    fn authority(
        partition: &'static str,
        origin: &'static str,
        database: &'static str,
        ephemeral: bool,
    ) -> BrowserAuthority {
        BrowserAuthority {
            partition,
            origin,
            database,
            ephemeral,
        }
    }

    #[test]
    fn browser_forged_and_revoked_grants_resolve_to_nothing() {
        let mut table = OpaqueGrantTable::default();
        table
            .register(
                "ca11ab1e",
                authority("personal", "https://a.example", "app", false),
            )
            .unwrap();

        assert!(table.resolve("forged").is_none());
        assert!(table.revoke("ca11ab1e").is_some());
        assert!(table.resolve("ca11ab1e").is_none());
    }

    #[test]
    fn browser_cross_scope_and_residency_checks_fail_closed() {
        let mut table = OpaqueGrantTable::default();
        table
            .register(
                "ca11ab1e",
                authority("personal", "https://a.example", "app", false),
            )
            .unwrap();

        for expected in [
            authority("personal", "https://b.example", "app", false),
            authority("work", "https://a.example", "app", false),
            authority("personal", "https://a.example", "other", false),
            authority("personal", "https://a.example", "app", true),
        ] {
            assert!(table
                .resolve_matching("ca11ab1e", |actual| actual == &expected)
                .is_none());
        }

        assert_eq!(
            table.resolve("ca11ab1e"),
            Some(&authority("personal", "https://a.example", "app", false))
        );
    }

    #[test]
    fn browser_duplicate_registration_never_rebinds_existing_authority() {
        let first = authority("personal", "https://a.example", "app", false);
        let replacement = authority("work", "https://b.example", "app", true);
        let mut table = OpaqueGrantTable::default();
        table.register("ca11ab1e", first.clone()).unwrap();

        assert_eq!(
            table.register("ca11ab1e", replacement),
            Err(OpaqueGrantTableError::DuplicateToken)
        );
        assert_eq!(table.resolve("ca11ab1e"), Some(&first));
    }

    #[test]
    fn browser_partition_destruction_revokes_only_matching_ephemeral_grants() {
        let mut table = OpaqueGrantTable::default();
        table
            .register(
                "ephemeral-a",
                authority("private", "https://a.example", "app", true),
            )
            .unwrap();
        table
            .register(
                "ephemeral-b",
                authority("other", "https://a.example", "app", true),
            )
            .unwrap();
        table
            .register(
                "persistent-a",
                authority("private", "https://a.example", "app", false),
            )
            .unwrap();

        let revoked = table.revoke_where(|grant| grant.ephemeral && grant.partition == "private");

        assert_eq!(revoked, 1);
        assert!(table.resolve("ephemeral-a").is_none());
        assert!(table.resolve("ephemeral-b").is_some());
        assert!(table.resolve("persistent-a").is_some());
    }
}
