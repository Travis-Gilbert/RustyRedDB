//! Frozen desktop-shell contracts for `SPEC-THEOREM-DESKTOP-SHELL-SERVO-RR-1.0`.
//!
//! W00 freezes the five vocabularies every desktop lane shares: stable
//! identity, one-writer RR ownership, the multi-parent surface binding,
//! origin-scoped storage grants, and replication policy. The lanes that follow
//! consume these types; they do not redefine them.
//!
//! This module is append-only. A breaking change mints
//! `theorem-desktop-surface/2` and retains the v1 decoder until usage and
//! rollback receipts permit removal.
//!
//! Three names were deliberately not reused, and the reasons are part of the
//! contract:
//!
//! - [`DesktopSurfaceId`] is not `agpui_shell_protocol::SurfaceId`. That type
//!   means "one mount root inside a plugin instance"; this one means "one slot
//!   in the workbench". [`AgpuiMount`] is the single place they meet.
//! - [`DesktopWindowId`] is not `gpui::WindowId` or `winit::window::WindowId`.
//!   Both already exist in this tree, and a third bare `WindowId` would be
//!   ambiguous at every import site.
//! - [`ReplicationPolicy`] is not `commonplace::Residency`. That type
//!   classifies a CommonPlace `Item` and its `Hosted` arm means the body lives
//!   in the cloud. A desktop root is a different subject, and `ConnectedLive`
//!   has no `Residency` counterpart. [`ReplicationPolicy::residency`] is the
//!   one mapping point so the vocabularies cannot drift apart.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Plugin identity stored inside a desktop root.
///
/// A string on the wire, matching the product protocol's plugin id, without
/// taking a dependency on that protocol.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct PluginId(pub String);

/// Mount root inside a plugin instance.
///
/// A string on the wire, matching the product protocol's surface id, without
/// taking a dependency on that protocol.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AgpuiSurfaceId(pub String);

/// The frozen contract slug. Mirrors `theorem-desktop-surface-v1.json`.
pub const DESKTOP_SURFACE_CONTRACT: &str = "theorem-desktop-surface/1";

/// Mint a desktop identity newtype with a routable, prefix-tagged stable key.
///
/// Every desktop identity is an opaque string with one URL-safe encoding, so
/// persistence and routing cannot disagree about what a key means.
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
    /// One GPUI window in the Theorem process. Distinct from `gpui::WindowId`,
    /// which is a renderer handle that does not survive a restart.
    DesktopWindowId,
    "window:"
);
desktop_id!(
    /// One restorable workspace: rail, dock layout and active surfaces.
    WorkspaceId,
    "workspace:"
);
desktop_id!(
    /// One registered application contributing an `AppDescriptor`.
    AppId,
    "app:"
);
desktop_id!(
    /// One hosted slot in the workbench: a Servo view, a terminal, an agent
    /// pane or a plugin panel. Not an AGPUI mount root; The product shell maps it onto a mount root.
    DesktopSurfaceId,
    "surface:"
);
desktop_id!(
    /// One native parent that surfaces attach into, in practice one per GPUI
    /// window. Many surfaces share a parent; one Servo engine serves them all
    /// (O16).
    ParentId,
    "parent:"
);
desktop_id!(
    /// One persistent browser storage partition. Ephemeral partitions are
    /// addressed by this type too, and destroyed rather than admitted (O04).
    BrowserPartitionId,
    "partition:"
);
desktop_id!(
    /// One browsing profile. Groups partitions for presentation; it never
    /// widens a storage grant.
    ProfileId,
    "profile:"
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

/// A root inside the desktop database, named so its policy is a property of
/// the root rather than a per-call argument.
///
/// There is no policy setter anywhere in this contract. That is what makes
/// O05's "policy mutation refusal from JavaScript" structural instead of
/// enforced by a guard that someone can forget to call.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DesktopRoot {
    /// Rail, workspace and dock layout shared across devices.
    WorkbenchShared { workspace: WorkspaceId },
    /// Redacted `BrowserCompatibilityOutcome` records (O09).
    CompatibilityOutcomes,
    /// Project entities and membership.
    Projects,
    /// Configuration an admitted plugin persists through its granted app lens.
    PluginConfig { plugin: PluginId },
    /// Window position and size. Device-shaped, so it never replicates.
    WindowGeometry {
        device: DeviceId,
        window: DesktopWindowId,
    },
    /// Scratch state discarded on clean shutdown.
    Ephemeral { device: DeviceId },
    /// Arbitrary third-party web storage. Durable locally, never admitted.
    ThirdPartyStorage { partition: BrowserPartitionId },
    /// A remote surface read live rather than replicated.
    ConnectedRemote { surface: DesktopSurfaceId },
}

impl DesktopRoot {
    /// The replication policy this root carries. Total, and the single source
    /// for both the positive and negative assertions in O05.
    #[must_use]
    pub const fn policy(&self) -> ReplicationPolicy {
        match self {
            Self::WorkbenchShared { .. }
            | Self::CompatibilityOutcomes
            | Self::Projects
            | Self::PluginConfig { .. } => ReplicationPolicy::Synced,
            Self::WindowGeometry { .. }
            | Self::Ephemeral { .. }
            | Self::ThirdPartyStorage { .. } => ReplicationPolicy::LocalOnly,
            Self::ConnectedRemote { .. } => ReplicationPolicy::ConnectedLive,
        }
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
    /// The replication policy of the root this grant writes into. Third-party
    /// web storage is durable locally and never admitted to synchronization.
    #[must_use]
    pub fn policy(&self) -> ReplicationPolicy {
        match self.residency {
            PartitionResidency::Ephemeral => ReplicationPolicy::LocalOnly,
            PartitionResidency::Persistent => DesktopRoot::ThirdPartyStorage {
                partition: self.scope.partition.clone(),
            }
            .policy(),
        }
    }
}

/// One surface's placement: which window shows it, which native parent it
/// attaches into, and which application owns it.
///
/// Many bindings share one [`ParentId`], and many parents share one Servo
/// engine. That is the whole multi-window contract (O16).
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceBinding {
    pub surface: DesktopSurfaceId,
    pub parent: ParentId,
    pub window: DesktopWindowId,
    pub app: AppId,
}

/// The one place a workbench slot and a plugin mount root meet.
///
/// [`AgpuiMount::surface`] names the workbench slot. [`AgpuiMount::mount`]
/// names the mount root inside the plugin instance. The two strings stay
/// distinct on the wire.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgpuiMount {
    pub surface: DesktopSurfaceId,
    pub plugin: PluginId,
    pub mount: AgpuiSurfaceId,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_desktop_identity_is_a_bare_string_on_the_wire() {
        let wire = serde_json::to_string(&DesktopSurfaceId::new("browser-google")).unwrap();
        assert_eq!(wire, r#""browser-google""#);
    }

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
    fn a_stable_key_refuses_a_foreign_prefix_instead_of_defaulting() {
        assert_eq!(
            DesktopSurfaceId::from_stable_key("partition:compat-matrix"),
            None
        );
        assert_eq!(ParentId::from_stable_key("browser-google"), None);
    }

    #[test]
    fn every_minted_identity_carries_a_distinct_prefix() {
        let prefixes = [
            DatabaseId::prefix(),
            DeviceId::prefix(),
            DesktopWindowId::prefix(),
            WorkspaceId::prefix(),
            AppId::prefix(),
            DesktopSurfaceId::prefix(),
            ParentId::prefix(),
            BrowserPartitionId::prefix(),
            ProfileId::prefix(),
            Origin::prefix(),
            StorageGrantId::prefix(),
        ];
        let mut seen = prefixes.to_vec();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), prefixes.len(), "prefixes collide: {prefixes:?}");
    }

    #[test]
    fn shared_workbench_state_is_admitted_and_device_state_is_not() {
        let shared = DesktopRoot::WorkbenchShared {
            workspace: WorkspaceId::new("default"),
        };
        let geometry = DesktopRoot::WindowGeometry {
            device: DeviceId::new("mac-studio"),
            window: DesktopWindowId::new("main"),
        };
        assert_eq!(shared.policy(), ReplicationPolicy::Synced);
        assert!(shared.policy().is_admitted());
        assert_eq!(geometry.policy(), ReplicationPolicy::LocalOnly);
        assert!(!geometry.policy().is_admitted());
    }

    #[test]
    fn arbitrary_third_party_storage_is_never_admitted() {
        let root = DesktopRoot::ThirdPartyStorage {
            partition: BrowserPartitionId::new("default"),
        };
        assert_eq!(root.policy(), ReplicationPolicy::LocalOnly);
        assert!(!root.policy().is_admitted());
    }

    #[test]
    fn a_connected_root_stays_remote_and_has_no_residency_counterpart() {
        let root = DesktopRoot::ConnectedRemote {
            surface: DesktopSurfaceId::new("chatgpt"),
        };
        assert_eq!(root.policy(), ReplicationPolicy::ConnectedLive);
        assert!(!root.policy().is_admitted());
        assert_eq!(root.policy().residency(), None);
        assert_eq!(ReplicationPolicy::LocalOnly.residency(), Some("local"));
        assert_eq!(ReplicationPolicy::Synced.residency(), Some("synced"));
    }

    #[test]
    fn a_desktop_root_tags_itself_in_snake_case() {
        let wire = serde_json::to_value(DesktopRoot::CompatibilityOutcomes).unwrap();
        assert_eq!(wire["kind"], "compatibility_outcomes");
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

    #[test]
    fn many_surfaces_share_one_parent_and_two_windows_do_not() {
        let parent = ParentId::new("main-content");
        let google = SurfaceBinding {
            surface: DesktopSurfaceId::new("google"),
            parent: parent.clone(),
            window: DesktopWindowId::new("main"),
            app: AppId::new("theorem.browser"),
        };
        let chatgpt = SurfaceBinding {
            surface: DesktopSurfaceId::new("chatgpt"),
            parent: parent.clone(),
            window: DesktopWindowId::new("main"),
            app: AppId::new("theorem.browser"),
        };
        let secondary = SurfaceBinding {
            surface: DesktopSurfaceId::new("claude"),
            parent: ParentId::new("secondary-content"),
            window: DesktopWindowId::new("secondary"),
            app: AppId::new("theorem.browser"),
        };
        assert_eq!(google.parent, chatgpt.parent);
        assert_ne!(google.parent, secondary.parent);
        assert_ne!(google.window, secondary.window);
    }

    #[test]
    fn a_workbench_slot_and_an_agpui_mount_root_stay_separate_identities() {
        let mount = AgpuiMount {
            surface: DesktopSurfaceId::new("records-panel"),
            plugin: PluginId("theorem.records".to_owned()),
            mount: AgpuiSurfaceId("composition".to_owned()),
        };
        let wire = serde_json::to_value(&mount).unwrap();
        assert_eq!(wire["surface"], "records-panel");
        assert_eq!(wire["mount"], "composition");
        assert_ne!(wire["surface"], wire["mount"]);
    }

    #[test]
    fn the_contract_slug_is_the_frozen_v1_name() {
        assert_eq!(DESKTOP_SURFACE_CONTRACT, "theorem-desktop-surface/1");
    }
}
