# Compatibility

- Crate version and file format version are different numbers.
- A rename of a Rust package does not migrate data.
- This beta reads and writes format 1 (`THEORRDB`, 4096-byte header).
- Older public Rusty Red distributions are different products. This release does not import them.
- The directory journal can be copied into `.RR` with `Database::migrate_journal`. That is the supported migration.
- Unknown capability versions can be opened read-only. Writable open refuses a capability this build does not implement (`rr_capability_unsupported`).
- Browser OPFS support remains behind the `browser-opfs` feature of `rustyred-log` and is not a native `.RR` file.
- Install from this git repository. `cargo package -p rustyred --locked --allow-dirty --list` lists the facade sources and exits 0. The facade still path-depends on sibling crates in this repository. Those crates are not on crates.io, and this beta does not upload them.
