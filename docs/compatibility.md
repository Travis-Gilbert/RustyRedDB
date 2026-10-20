# Compatibility

- Crate version and file format version are different numbers.
- A rename of a Rust package does not migrate data.
- This beta reads and writes format 1 (`THEORRDB`, 4096-byte header).
- Older public Rusty Red distributions are different products. This release does not import them.
- The directory journal can be copied into `.RR` with `Database::migrate_journal`. That is the supported migration.
- Unknown capability versions can be opened read-only. Writable open refuses a capability this build does not implement (`rr_capability_unsupported`).
- Browser OPFS support remains behind the `browser-opfs` feature of `rustyred-log` and is not a native `.RR` file.
- Published beta.1 remains available from Git. The beta.2 candidate is not yet on crates.io. Its versioned support forks and normalized package family are rehearsed through Cargo directory-source resolution; see [release preparation](releasing.md). This rehearsal does not satisfy the post-publication registry install gate.

The default facade uses the RedCore graph journal. Physical operation logs and their fold APIs remain explicit directory-backed advanced APIs, matching their existing deployment topology; they are not migrated or replaced by graph snapshots. See [storage owners](extraction/storage-owners.md).

The beta.2 contract crate owns only shared database/browser-storage identities. Theorem keeps product window, plugin, workspace and shell policies and reexports the shared storage types. Identical serde shapes alone are not treated as proof of Rust type identity.

The reviewed implementation pin is `06acfc450d9c56067cac68d1f0866047a81cfb45`.
The public Git consumer has an actual two-test runtime receipt at that pin.
Complete normalized archive/optimized CLI bundles are still pending the
[exact-source manual qualification](https://github.com/Travis-Gilbert/RustyRedDB/actions/runs/39506823736);
the workflow bootstrap revision is `0bc679df1653ed85d1afb33093c830e27e634f35`.
Workflow/doc attestation revisions do not change the library implementation pin.
