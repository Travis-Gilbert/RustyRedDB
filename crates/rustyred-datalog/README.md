# Theorem Datalog

One pure engine interface serves graph adapters and compute-offload. This crate
does not read the graph, network or filesystem. Callers supply immutable,
content-addressed data and retain the resulting derivation receipts.

`Compiled` runs the existing fixed rules with Datafrog and reports
`rust-datafrog-core`. `Runtime` parses caller-supplied scalar relational rules
through theorem-egraph and reports `egraph-theorem`. Runtime programs cannot
create equality sorts, union values or execute external commands.

```rust
use datalog::{Engine, FactPack, Runtime};
use serde_json::json;
use std::collections::BTreeMap;

let engine = Runtime::new(r#"
    (relation edge (String String))
    (relation path (String String))
    (rule ((edge a b)) ((path a b)) :name "base")
    (rule ((path a b) (edge b c)) ((path a c)) :name "transitive")
"#)?;
let facts = FactPack::new(BTreeMap::from([("edge".into(), vec![
    vec![json!("A"), json!("B")], vec![json!("B"), json!("C")],
])]));
let receipt = engine.derive(&facts);
receipt.ensure_complete()?;
assert!(receipt.derived.iter().any(|row| row.tuple == vec![json!("A"), json!("C")]));
# Ok::<(), datalog::Error>(())
```

Runtime supports stratified relation negation such as `(not (edge a c _))`.
It runs the canonical evaluator's anti-join after the referenced lower stratum
reaches a fixed point. Negative variables must be bound; `_` is an anonymous
column. Recursive cycles through negation are refused. Positive recursion stays
supported, with one shared iteration budget across all strata.

Every runtime consequence identifies actual matched premises. Literal program
facts identify their rule pack; they are not invented external evidence. An
iteration limit or malformed input never becomes a successful empty result.
Negative premises additionally carry `AbsenceEvidence`: relation, bound/wildcard
pattern, closed-relation hash, evaluator-input hash, caller `fact_pack` hash,
`rule_pack` hash, and stratum. A graph publisher must retain this explicit
closed-input assumption; it cannot promote scoped absence to global absence.
Empty negative-premise lists are omitted on the wire for compatibility with
existing positive receipts.
Wall time is an observation and must not be used as CPU time or an estimate.

Compiled legacy rows have four columns: entity ID, attributes object, source
reference and fact ID. `fact_pack_from_legacy_rows` adapts the existing Python
wire records. `derive_legacy_receipt` preserves the historical Python receipt
schema and engine string; new consumers use the typed `DerivationReceipt`.

The unchanged Python reference supplies 45 regression fixtures across its
fourteen current rules. The retained native-only `same_entity` rule has a
separate Rust provenance test. Neither substitutes for the actual native.py
extension parity gate run by the application.
