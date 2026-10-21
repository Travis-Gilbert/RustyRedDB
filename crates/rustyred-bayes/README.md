# rustyred-bayes

Small Bayesian posterior values, deterministic draws, and content-addressed receipts. The library import remains `theorem_bayes`.

The crate intentionally stays light: serde shapes, conjugate posterior updates,
deterministic seeded draws for replay, and content-addressed posterior receipts.
It does not pull PyMC, NUTS, or a heavy runtime sampler into the Rust hot path.

`fold_binary_receipts` accepts recursive receipt trees but traverses them with an
explicit stack plus node/depth limits. The checked oracle fixture lives at
`tests/fixtures/affordance_posterior_oracle.json` and was captured from the source baseline. The package includes the fixture; it does not require the original application or Python generator.

From the public RustyRedDB repository root:

```sh
cargo +1.96.1 test --locked -p rustyred-bayes
```
