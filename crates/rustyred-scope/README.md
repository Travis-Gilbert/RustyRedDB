# rustyred-scope

`CapabilityScope` is a dependency-light intersection over four string allowlists.
The Rust import remains `theorem_capability_scope`; its serialized fields stay
unchanged. The crate only depends on serde and does not load a graph, provider,
application, or agent runtime.

From the public RustyRedDB repository root, run
`cargo +1.96.1 test --locked -p rustyred-scope`.

## The inversion worth knowing

**An all-empty scope is unrestricted.** Every populated dimension is an
intersection constraint, so `allow_servers = ["github"]` with
`allow_tags = ["write"]` admits github write tools and nothing else — while
`CapabilityScope::default()` admits everything.

That is why "this person turned every capability off" is `None` and never
`CapabilityScope::default()`: a turn with an empty dial must reach *nothing*,
and the default reaches *everything*. `is_unrestricted()` exists so callers can
say which one they mean.

## Implementing `Admissible`

Four accessors rather than the whole affordance record, because those are the
four dimensions the intersection reads; every other field on an affordance is
graph-shaped and this crate does not know about graphs.

```rust
use theorem_capability_scope::{Admissible, CapabilityScope};

struct Tool { id: String, family: String, tags: Vec<String> }

impl Admissible for Tool {
    fn scope_affordance_id(&self) -> &str { &self.id }
    fn scope_server_id(&self) -> &str { "" } // built in, no server
    fn scope_family(&self) -> &str { &self.family }
    fn scope_tags(&self) -> &[String] { &self.tags }
}

let scope = CapabilityScope {
    allow_families: vec!["graph".to_owned()],
    ..CapabilityScope::default()
};
let recall = Tool { id: "recall".into(), family: "graph".into(), tags: vec![] };
assert!(scope.admits(&recall));
```
