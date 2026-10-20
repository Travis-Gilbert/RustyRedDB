# Release identity

- Product: Rusty Red
- GitHub: https://github.com/Travis-Gilbert/RustyRedDB
- Public library crate: `rustyred`
- Command: `rustyred`
- Version: `0.1.0-beta.1`
- License: MIT for this extraction. Vendored `egg` and `turbovec` stay MIT. `rustyred-datalog` and `rustyred-egraph` stay MIT OR Apache-2.0.
- crates.io names `rustyred`, `rustyred-engine`, `rustyred-query`, `rustyred-log`, `rustyred-cli`, and `rustyred-db` were unclaimed on the sparse index on 2026-10-20 (HTTP 404). That is not a reservation. This repository does not publish to crates.io.
- File format version is independent of the crate version: magic `THEORRDB`, format `1`, header `4096` bytes.
- Rust library names inside the engine crates are unchanged (`rustyred_thg_core`, `theorem_oplog`, `theorem_surface_contracts`, and the rest) so serialized code and Theorem imports keep one type identity.
