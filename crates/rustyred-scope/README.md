# theorem-capability-scope

`CapabilityScope`: the subset of affordances one agent, or one turn, is primed
for. Serde and four string lists, nothing else.

## Why it is its own crate

The type used to live in `rustyred-thg-affordances`, which reaches the graph
store, the plugin host, and the harness kernel.

`SPEC-THEOREMWEB-COMPOSER-1.0` ADDENDUM-2 (A9/D10) binds the composer's
capability dial to *this* type — "re-exported, not mirrored" — so that a dial
entry and a run's affordance gate cannot drift into two vocabularies. But
`theorem-surface-contracts` compiles into a `wasm32-unknown-unknown` leaf and
is dependency-light on purpose, so it cannot take an edge to the affordance
crate. Mirroring the type is what the addendum forbids.

So the charter moved down to a leaf that only ever needed `serde`, and both
sides re-export it:

```
theorem-surface-contracts ─┐
                           ├─→ theorem-capability-scope
rustyred-thg-affordances ──┘
```

Every existing `rustyred_thg_affordances::CapabilityScope` path still resolves,
and every existing `scope.admits(&affordance)` call still compiles, because
`admits` is an inherent method generic over the local `Admissible` trait rather
than a trait method callers would have to import.

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
