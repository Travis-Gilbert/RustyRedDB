# Preserved storage owners

The source has two explicit storage paths. Extraction does not convert either into the other.

| Boundary | Canonical implementation | Persistence | Consumers |
| --- | --- | --- | --- |
| Embedded graph facade | `rustyred-engine::RedCoreGraphStore` | Immutable chunks plus RootCatalog in directory `redcore.journal` or native `.RR` body | `rustyred::Database`, CLI, Theorem embedded Engine |
| Physical operation ledger and derived folds | `rustyred-log::PhysicalLog`, `ProposalCommitter`, `rustyred-engine::FoldGraphStore`, `FoldRuntime` | Existing physical log directory and explicit fold storage floors | Theorem record-form runtime/memtensor, harness algebra hooks and scheduling |

At the plan pin, embedded `Engine` already called `RedCoreGraphStore::open`; it did not open `PhysicalLog`. The `graph/fold_store.rs` implementation is byte-identical at the plan pin, Cursor extraction source, and original OSS library commit (SHA-256 `0c629089762672a7062ce58584b055b4f017fe2ef613529ab600194d1d5993b3`). The physical crate root is still `src/physical_lib.rs`, not the older graph-backed product ledger `src/lib.rs`.

Theorem's `theorem-oplog-graph-compat` owns that older product projection and is retained downstream. It is not a second copy of the generic physical log. The cutover gives its source its own crate directory instead of pointing into the physical-log compatibility shim.

The `.RR` single-file contract covers the graph facade's graph, relationships, designated supplied-vector payloads, file payloads, retained graph revisions and catalog. The explicit physical operation-log/fold API retains its directory topology; it is not silently represented by a graph snapshot or advertised as a single-file ledger. Existing directory/log deployments are not migrated.

The focused generic oracles are `graphstore_fold_differential`, `fold_algebra_laws`, physical-log tests and the public `.RR` roundtrip/failure tests. Theorem's actual downstream consumer tests must run on the same candidate pin before G10 passes. Matching source hashes alone is not a passing runtime gate.
