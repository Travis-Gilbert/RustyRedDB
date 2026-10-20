# rustyred-egraph

The pure symbolic engine, imported as `egraph`. It owns the pinned
`egglog = 2.0.0` dependency and has no graph-store or product dependencies.

`Engine` loads a content-addressed `RulePack`, executes facts, runs a bounded
schedule, and extracts a term using declared or caller-supplied additive costs.
`Term.expression` preserves the pinned engine's exact renderer output. Invalid
packs and unresolved roots return errors. Receipts identify the program and
cost model, report actual evaluator iterations and matches, and record elapsed
time as an observation. An iteration limit is distinct from a fixed point.

`RelationalEngine` accepts scalar relation declarations and stratified relational
rules through egglog's parsed AST. Columns are strings, signed 64-bit integers,
finite floats, or booleans. It refuses user-defined equality sorts, unions,
rewrites, arbitrary functions, destructive actions, and external I/O before
execution. Scalar equality tests and bindings do not introduce an equality
sort. Each invocation starts from the validated program baseline.

Relational provenance comes from typed witness tables inserted into the actual
executed rule heads. Each witness records a produced tuple and the positive
premise tuples that matched. A body clause `(not (edge a c _))` tests absence
only after the referenced relation's lower stratum reaches a fixed point.
The pinned evaluator builds a closed projection using Set union and evaluates
`set-not-contains`; Rust only encodes scalar tuple keys. `_` is independently
anonymous in a relation column. All named negative variables must be grounded
by positive columns or scalar equality bindings. Negative dependency cycles,
unsafe variables, and anonymous head/scalar expressions are refused.

Each matched negative premise retains the relation, bound columns (`None` for
wildcards), hash of the closed lower relation, canonical input-facts hash,
rule-pack hash, and consuming stratum. These are closed-input assumptions;
they do not prove absence in a graph or deployment that the caller did not
supply. The single iteration budget covers lower closure, Set projection, and
upper evaluation. Exhausting it returns `IterationLimit` and never evaluates
a higher stratum against an incomplete lower relation. Evaluator counts include
the actual internal witness and Set projection rules.

The crate does not choose application rule packs, load tenant graphs, or write
receipts into graph storage. Consumers retain those responsibilities.

```rust
use egraph::{CostModel, Engine, RulePack, Schedule};

let pack = RulePack::new(
    "(datatype T (Num i64) (Add T T))
     (rewrite (Add x (Num 0)) x)",
    CostModel::default(),
);
let mut engine = Engine::new(pack)?;
let root = engine.root("(Add (Num 7) (Num 0))")?;
let saturation = engine.saturate(Schedule::FixedPoint {
    rulesets: vec![String::new()], max_iterations: 16,
})?;
let (term, receipt) = engine.extract(root, &CostModel::default())?;
assert_eq!(term.expression, "(Num 7)");
# Ok::<(), egraph::Error>(())
```

Validate with `cargo +1.96.1 test --locked -p rustyred-egraph`. The native tests compare extraction
and counters with the pinned engine directly, exercise real relational witness
proofs, and verify refusal, invocation isolation, and iteration limits.
