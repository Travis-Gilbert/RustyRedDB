//! Canonical database identity, writer ownership and browser storage contracts.
//! Product window, plugin, workspace and surface policies remain downstream.

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! desktop_id {
    ($(#[$meta:meta])* $name:ident, $prefix:literal) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            #[must_use]
            pub fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }

            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// The prefix this identity carries inside a stable key.
            #[must_use]
            pub const fn prefix() -> &'static str {
                $prefix
            }

            /// URL-safe routable key, prefix included.
            #[must_use]
            pub fn stable_key(&self) -> String {
                format!("{}{}", $prefix, self.0)
            }

            /// Inverse of [`Self::stable_key`]. Refuses an unrecognised prefix
            /// instead of defaulting, because silently landing on a different
            /// object is worse than refusing the key.
            #[must_use]
            pub fn from_stable_key(key: &str) -> Option<Self> {
                key.strip_prefix($prefix).map(|rest| Self(rest.to_owned()))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }
    };
}

desktop_id!(
    /// One native `.RR` file. Stable across copy, move and rename, so copying a
    /// cleanly closed file preserves logical identity (O14).
    DatabaseId,
    "rr:"
);
desktop_id!(
    /// One installation on one machine. Names the RR write owner and scopes
    /// device-local roots that must never replicate (O05).
    DeviceId,
    "device:"
);
desktop_id!(
    /// One persistent browser storage partition. Ephemeral partitions are
    /// addressed by this type too, and destroyed rather than admitted (O04).
    BrowserPartitionId,
    "partition:"
);
desktop_id!(
    /// A web origin in serialized form, e.g. `https://example.com`.
    Origin,
    "origin:"
);
desktop_id!(
    /// An opaque capability handle for one storage scope.
    ///
    /// The value carries no origin, partition or database name. A holder
    /// cannot widen its own reach by editing the token, and a forged value
    /// resolves to nothing (O04).
    StorageGrantId,
    "grant:"
);

/// Where a desktop root is allowed to travel.
///
/// O05 pins these three names. See the module docs for why this is not
/// `commonplace::Residency`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplicationPolicy {
    /// Never leaves this device, even when the database is synchronized.
    LocalOnly,
    /// Admitted to synchronization through the existing pack, diff, merge, CAS
    /// and receipt machinery. There is no browser-specific replication path.
    Synced,
    /// Lives remotely and is read live. Synchronization never materializes it
    /// into the local durable root.
    ConnectedLive,
}

impl ReplicationPolicy {
    /// The `commonplace::Residency` label this policy maps onto, or `None` when
    /// there is no counterpart. This is the only bridge between the two
    /// vocabularies.
    #[must_use]
    pub const fn residency(self) -> Option<&'static str> {
        match self {
            Self::LocalOnly => Some("local"),
            Self::Synced => Some("synced"),
            Self::ConnectedLive => None,
        }
    }

    /// Whether synchronization may carry this root.
    #[must_use]
    pub const fn is_admitted(self) -> bool {
        matches!(self, Self::Synced)
    }
}

/// How a process holds the single `.RR` file.
///
/// Exactly one writer exists per database. A second writable open is refused by
/// name rather than blocked, so the caller learns which device holds the lock
/// (O02).
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RrAccess {
    /// The sole write owner. Servo, plugins and secondary windows never hold
    /// this; they reach storage through a broker.
    Writer { device: DeviceId },
    /// A concurrent reader. Any number may exist beside one writer.
    ReadOnly,
}

/// One process's claim on one database.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RrOwnership {
    pub database: DatabaseId,
    pub access: RrAccess,
}

impl RrOwnership {
    #[must_use]
    pub const fn is_writer(&self) -> bool {
        matches!(self.access, RrAccess::Writer { .. })
    }
}

/// The three-part key every browser storage operation is scoped by (O04).
///
/// Two origins in one partition, and one origin across two partitions, are
/// distinct scopes. Clearing either leaves the other untouched.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageScope {
    pub partition: BrowserPartitionId,
    pub origin: Origin,
    pub database: String,
}

/// Whether a partition's state outlives the session.
#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PartitionResidency {
    /// Written to the durable browser root under the declared policy.
    #[default]
    Persistent,
    /// Destroyed on close and never admitted to the durable root.
    Ephemeral,
}

/// A capability to touch exactly one storage scope.
///
/// The broker issues the grant; the holder presents only [`StorageGrant::id`].
/// The scope stays on the broker side, which is why a forged id reaches
/// nothing.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageGrant {
    pub id: StorageGrantId,
    pub scope: StorageScope,
    pub residency: PartitionResidency,
}

impl StorageGrant {
    /// Browser payload roots are device-local, for both residency modes.
    #[must_use]
    pub fn policy(&self) -> ReplicationPolicy {
        ReplicationPolicy::LocalOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_stable_key_round_trips_through_its_own_prefix() {
        let id = BrowserPartitionId::new("compat-matrix");
        assert_eq!(id.stable_key(), "partition:compat-matrix");
        assert_eq!(
            BrowserPartitionId::from_stable_key(&id.stable_key()),
            Some(id)
        );
    }

    #[test]
    fn exactly_one_access_shape_is_a_writer() {
        let writer = RrOwnership {
            database: DatabaseId::new("travis"),
            access: RrAccess::Writer {
                device: DeviceId::new("mac-studio"),
            },
        };
        let reader = RrOwnership {
            database: DatabaseId::new("travis"),
            access: RrAccess::ReadOnly,
        };
        assert!(writer.is_writer());
        assert!(!reader.is_writer());
    }

    #[test]
    fn a_storage_scope_separates_two_origins_and_two_partitions() {
        let base = StorageScope {
            partition: BrowserPartitionId::new("default"),
            origin: Origin::new("https://a.example"),
            database: "notes".to_owned(),
        };
        let other_origin = StorageScope {
            origin: Origin::new("https://b.example"),
            ..base.clone()
        };
        let other_partition = StorageScope {
            partition: BrowserPartitionId::new("compat-matrix"),
            ..base.clone()
        };
        assert_ne!(base, other_origin);
        assert_ne!(base, other_partition);
        assert_ne!(other_origin, other_partition);
    }

    #[test]
    fn an_ephemeral_grant_is_local_only_and_a_persistent_one_is_too() {
        let scope = StorageScope {
            partition: BrowserPartitionId::new("default"),
            origin: Origin::new("https://example.com"),
            database: "idb".to_owned(),
        };
        let ephemeral = StorageGrant {
            id: StorageGrantId::new("01J000"),
            scope: scope.clone(),
            residency: PartitionResidency::Ephemeral,
        };
        let persistent = StorageGrant {
            id: StorageGrantId::new("01J001"),
            scope,
            residency: PartitionResidency::Persistent,
        };
        assert_eq!(ephemeral.policy(), ReplicationPolicy::LocalOnly);
        assert_eq!(persistent.policy(), ReplicationPolicy::LocalOnly);
        assert!(!persistent.policy().is_admitted());
    }

    #[test]
    fn a_grant_id_reveals_nothing_about_the_scope_it_unlocks() {
        let grant = StorageGrant {
            id: StorageGrantId::new("01J0ABCDEF"),
            scope: StorageScope {
                partition: BrowserPartitionId::new("default"),
                origin: Origin::new("https://example.com"),
                database: "idb".to_owned(),
            },
            residency: PartitionResidency::Persistent,
        };
        let wire = serde_json::to_string(&grant.id).unwrap();
        assert!(!wire.contains("example.com"));
        assert!(!wire.contains("default"));
        assert!(!wire.contains("idb"));
    }
}

#[cfg(test)]
mod wire_identity_parity {
    use super::*;

    #[test]
    fn storage_identity_wire_preimages_stay_frozen() {
        assert_eq!(
            serde_json::to_string(&DatabaseId::new("fixture")).unwrap(),
            "\"fixture\""
        );
        assert_eq!(DatabaseId::new("fixture").stable_key(), "rr:fixture");
        assert_eq!(DeviceId::new("fixture").stable_key(), "device:fixture");
        assert_eq!(
            Origin::new("https://example.test").stable_key(),
            "origin:https://example.test"
        );
        assert_eq!(DatabaseId::from_stable_key("device:fixture"), None);
        assert_eq!(
            serde_json::to_value(RrAccess::ReadOnly).unwrap(),
            serde_json::json!({"kind":"read_only"})
        );
        assert_eq!(
            serde_json::to_value(RrAccess::Writer {
                device: DeviceId::new("fixture")
            })
            .unwrap(),
            serde_json::json!({"kind":"writer","device":"fixture"})
        );
    }
}
