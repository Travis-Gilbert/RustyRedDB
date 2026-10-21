# rustyred-contracts

Dependency-light database/sync/storage/link contracts. The Rust library import
remains `theorem_surface_contracts`; serialized names and shared type identities
are preserved. The retained modules are `article`, `article_links`, `desktop`
(neutral storage ownership types), `link`, `link_api`, `sync` and `wiki`.

This crate contains no UI shell, application registry or database owner. Higher
layers reexport the same types instead of defining mirrored copies. Serde and
TypeScript shape derivation remain available.

From the public RustyRedDB repository root:

```sh
TS_RS_EXPORT_DIR=/absolute/disposable-bindings \
  cargo +1.96.1 test --locked -p rustyred-contracts
```

The explicit output directory keeps generated test bindings outside reviewed
source. See [release preparation](../../docs/releasing.md) for compiler and
source-distribution requirements.
