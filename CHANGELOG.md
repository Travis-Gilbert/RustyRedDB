# Changelog

## 0.1.0-beta.2 (unpublished)

- Restore optional GraphBLAS and query-saturation modules and keep storage contracts independent of Theorem desktop types.
- Name the patched egg and turbovec support forks explicitly, retain their library APIs and encoding versions, and prepare normalized public packages without consumer patches or sibling source dependencies.
- Use typed Clap subcommands in `commands.rs`, with help/version output and rejection of unknown flags, missing values and invalid parameter objects before database access.
- Preserve indeterminate graph/vector and physical-log commit outcomes, refuse writes through poisoned handles, and distinguish already-durable fold commits from subsequent maintenance failures.
- Add native CI with read-only repository permissions, a Rust 1.96.1 compiler floor, default-profile outside consumers, dependency/license inventories, cold-target measurement tooling and publication-disabled source/optimized-CLI artifact preparation.
- Remove the publication-only derivation test cycle by retaining its 15 real graph-store tests in the engine integration suite; ordinary workspace package and publication dry runs now prepare all crates with their lockfiles.
- Prepare an opt-in registry-only post-publication consumer, example and CLI smoke check; its live result stays unverified until actual publication.
- Review the Theorem cutover separately against one immutable public implementation pin; no live database migration is part of this candidate.

The `.RR` format remains `THEORRDB`, version 1, with a 4096-byte header. Beta.1 history and its tag remain unchanged. This entry records implementation changes, not successful qualification of a future SHA. Final candidate receipts and its install pin follow validation; crates.io publication and registry-install gate G12 remain pending explicit authorization.

## 0.1.0-beta.1

First extractable beta of the embedded database. Native `.RR` create, open, read-only query, typed mutation, inspect, and verify. Directory journals still open. crates.io publication is not part of this tag.
