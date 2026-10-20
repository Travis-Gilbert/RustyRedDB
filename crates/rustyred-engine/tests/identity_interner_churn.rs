//! Growth oracle for the process-global identity table
//! (SPEC-THEOREM-MEMTENSOR-IDENTITY-INTERNER-1.0).
//!
//! The interner is append-only, so its retention is a stated contract rather
//! than an implementation detail: it holds one copy of each *distinct* admitted
//! id, and re-admitting the same row set across generations must intern
//! nothing. This binary holds exactly one test so that the global counts it
//! reads cannot race another test in the same process.

use rustyred_thg_core::intern::{intern, interner_stats, symbol_of};

#[test]
fn generations_over_a_stable_id_set_do_not_grow_the_table() {
    let ids = (0..256)
        .map(|index| format!("churn:node:{index}"))
        .collect::<Vec<_>>();

    let empty = interner_stats();
    assert_eq!(
        empty.symbols, 0,
        "the table starts empty in a fresh process"
    );

    let symbols = ids.iter().map(|id| intern(id)).collect::<Vec<_>>();
    let settled = interner_stats();
    assert_eq!(settled.symbols, ids.len());
    assert_eq!(
        settled.bytes,
        ids.iter().map(String::len).sum::<usize>(),
        "one copy of each id, and only one"
    );

    // Sixty-four generations of the same rows: this is the workload the
    // decision is for, where every generation used to allocate its own String
    // per row.
    for _ in 0..64 {
        for (id, expected) in ids.iter().zip(&symbols) {
            let symbol = intern(id);
            assert_eq!(symbol, *expected);
            assert_eq!(symbol.as_str(), id.as_str());
        }
    }
    assert_eq!(interner_stats(), settled, "re-admission interned nothing");

    // Misses are the other growth vector, and they are closed by construction:
    // a read path resolves through `symbol_of`, which never mints.
    for index in 0..1024 {
        assert!(symbol_of(&format!("churn:absent:{index}")).is_none());
    }
    assert_eq!(interner_stats(), settled, "lookups interned nothing");
}
