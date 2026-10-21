# Release identity

- Product: Rusty Red
- GitHub: https://github.com/Travis-Gilbert/RustyRedDB
- Public library crate: `rustyred`
- Command: `rustyred`
- Candidate version: `0.1.0-beta.2` (unpublished). Existing `v0.1.0-beta.1` remains immutable.
- License: MIT for this extraction. Vendored `egg` and `turbovec` stay MIT. `rustyred-datalog` and `rustyred-egraph` stay MIT OR Apache-2.0.
- All 15 publishable package identities were absent from the official crates.io sparse index on 2026-10-20 (HTTP 404): `rustyred`, `rustyred-bayes`, `rustyred-cli`, `rustyred-contracts`, `rustyred-datalog`, `rustyred-derivation`, `rustyred-egg`, `rustyred-egraph`, `rustyred-engine`, `rustyred-graphblas`, `rustyred-log`, `rustyred-query`, `rustyred-query-saturate`, `rustyred-scope`, and `rustyred-turbovec`. `parts-catalog` is a nonpublished workspace example. See the [registry-name receipt](distribution-evidence/registry-names-06acfc4.json). Absence is transient availability, not a reservation or proof of upload rights. No package has been published by this implementation.
- Before an authorized publication, the release owner must authenticate using Cargo's secure credential storage, repeat availability checks for all 15 names, inspect `cargo owner --list <package>` for any existing package, confirm the account/team has publication rights, and run the ordered package dry runs in the release guide. For a new name, ownership exists only after the first authorized successful publication. No ownership or credential mutation was performed during qualification.
- File format version is independent of the crate version: magic `THEORRDB`, format `1`, header `4096` bytes.
- Rust library names inside the engine crates are unchanged (`rustyred_thg_core`, `theorem_oplog`, `theorem_surface_contracts`, and the rest) so serialized code and Theorem imports keep one type identity.

- Support forks: `rustyred-egg` and `rustyred-turbovec`, version `0.1.0-beta.2`, retaining library names `egg` and `turbovec` and the extracted fixes. All names require the explicit ownership/availability procedure above before publication.
- Compiler floor for the candidate: Rust 1.96.1. Older inherited minimums were unqualified; no lower-version support is claimed.
- Distribution rehearsal and publication order: see `../releasing.md`. No package publication is authorized by implementation work.
