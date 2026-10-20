# Contributing

Issues and patches belong in this repository. Theorem product behavior stays in Theorem.

Run `cargo test -p rustyred --locked` and `python3 scripts/dependency_boundary.py` before sending a change that touches storage or the public API. Do not add a dependency on Theorem application crates.
