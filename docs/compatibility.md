# Compatibility

- Crate version and file format version are different numbers.
- A rename of a Rust package does not migrate data.
- This beta reads and writes format 1 (`THEORRDB`, 4096-byte header).
- Older public Rusty Red distributions are different products. This release does not import them.
- The directory journal can be copied into `.RR` with `Database::migrate_journal`. That is the supported migration.
- Unknown capability versions can be opened read-only. Writable open refuses a capability this build does not implement (`rr_capability_unsupported`).
- Browser OPFS support remains behind the `browser-opfs` feature of `rustyred-log` and is not a native `.RR` file.
- Published beta.1 remains available from Git. The beta.2 candidate is not yet on crates.io. Its versioned support forks and normalized package family are rehearsed through Cargo directory-source resolution; see [release preparation](releasing.md). This rehearsal does not satisfy the post-publication registry install gate.

`Database::create` and `Database::open` use RedCoreGraphStore with the native `.RR` chunk owner; graph state and application payloads remain in that single file. Only the explicit `create_directory` and `open_directory` APIs use `redcore.journal` in a directory. Physical operation logs and their fold APIs remain separate, explicit directory-backed advanced APIs, matching their existing deployment topology. See [storage owners](extraction/storage-owners.md).

The beta.2 contract crate owns only shared database/browser-storage identities. Theorem keeps product window, plugin, workspace and shell policies and reexports the shared storage types. Identical serde shapes alone are not treated as proof of Rust type identity.

The reviewed implementation pin is `984453329f99ae41a5d40e0297218b242ff5075a`.
Its public Git consumer passed both actual runtime tests under network denial.
The same candidate removes the derivation dev-dependency cycle by retaining all
15 real-store fold/why tests in engine's integration suite. Ordinary package
archives include Cargo.lock; clean workspace publication preparation passed its
nonuploading dry run. Both native normalized archive/optimized CLI bundles passed
[run 39509130713](https://github.com/Travis-Gilbert/RustyRedDB/actions/runs/39509130713)
and downloaded-byte review. All 15 crate archives match the clean local
publication-preparation bytes. See the [distribution record](extraction/distribution-review.md)
for actual Mac optimized CLI runtime, Linux binary inspection and separate
Linux native packaged CLI runtime evidence.
Registry publication and the post-publication G12 install are not claimed.
