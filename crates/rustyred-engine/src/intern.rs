//! Process-global node-identity interning for the MemTensor path.
//!
//! SPEC-THEOREM-MEMTENSOR-IDENTITY-INTERNER-1.0, Option A: identity is a
//! [`Symbol`] in memory and a `String` everywhere it crosses a durability or
//! content-addressing boundary. Nothing here is serialized: manifests, oplog
//! records, Algebra checkpoints, and `node_id()` results all stay string-keyed,
//! so no block content id, pin hash, or fold version moves.
//!
//! ## What the table costs, and what bounds it
//!
//! The table is append-only. A symbol, once minted, resolves for the life of
//! the process, because callers hold `Symbol` by value and
//! [`Symbol::as_str`] hands out a `&'static str` that a `MemTensorView` can
//! lend on through `node_id()`.
//!
//! That makes retention a real contract rather than an implementation detail,
//! so it is stated here:
//!
//! * The table retains one copy of each **distinct node id admitted into a
//!   fold or a loaded view** for the life of the process. It does not grow
//!   with the number of folds, views, generations, or reads over those ids -
//!   the second and later generations of the same row set intern nothing new.
//!   That is the case this replaces, where every generation held its own
//!   `String` per row.
//! * A **lookup never interns**. [`symbol_of`] answers from the table and
//!   returns `None` for an unknown id, so a query, a probe, or a miss on
//!   attacker-supplied input cannot grow it. Only admission - `ensure_node`
//!   on a fold, `load_memtensor` on a block - calls [`intern`].
//! * The table refuses rather than wraps when it exhausts the `u32` symbol
//!   space
//!   ([`try_intern`]); [`intern`] panics on that refusal because a caller past
//!   four billion distinct live identities has no recovery.
//!
//! The bound this does **not** give: a process that admits an unbounded stream
//! of *distinct* ids (per-request synthetic ids, a long-lived multi-tenant
//! server whose tenants never repeat a row) retains all of them. Reclamation
//! would mean either refcounted symbols (`node_id()` can no longer lend a
//! `&'static str`) or epoch-scoped tables (symbols stop being comparable
//! across folds). Both are behavior changes to the trait surface, so they are
//! a reopen trigger recorded in the spec, not something to guess at here.
//! [`interner_stats`] exists so that a caller can observe the table rather
//! than infer it.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

/// An interned node identity.
///
/// `Symbol` is `Copy` and four bytes. It deliberately implements neither `Ord`
/// nor `Display`: symbol values are assignment-ordered and process-local, so
/// any ordering or rendering must go through [`Symbol::as_str`] and the
/// canonical string.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Symbol(u32);

impl Symbol {
    /// Resolve to the canonical string.
    ///
    /// The borrow is `'static` because the table is append-only, which is what
    /// lets `MemTensorRead::node_id` keep returning a borrowed `&str`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        table().read().expect("identity interner poisoned").strings[self.0 as usize]
    }

    /// The raw symbol value. Process-local: never serialize it, never compare
    /// it across processes.
    #[must_use]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

/// Observed size of the process-global identity table.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InternerStats {
    /// Distinct identities interned since process start.
    pub symbols: usize,
    /// Bytes of identity text retained, excluding table overhead.
    pub bytes: usize,
}

/// Raised when the identity table cannot mint another symbol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InternerExhausted;

impl std::fmt::Display for InternerExhausted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("identity interner exhausted its u32 symbol space")
    }
}

impl std::error::Error for InternerExhausted {}

#[derive(Default)]
struct Table {
    symbols: HashMap<&'static str, Symbol>,
    strings: Vec<&'static str>,
    bytes: usize,
}

fn table() -> &'static RwLock<Table> {
    static TABLE: OnceLock<RwLock<Table>> = OnceLock::new();
    TABLE.get_or_init(RwLock::default)
}

/// Intern one identity, minting a symbol if this is the first sight of it.
///
/// # Panics
///
/// Panics if the table is exhausted. Use [`try_intern`] where a refusal is
/// preferable.
#[must_use]
pub fn intern(id: &str) -> Symbol {
    try_intern(id).expect("identity interner exhausted its u32 symbol space")
}

/// Intern one identity, refusing after the `u32` symbol space is full.
pub fn try_intern(id: &str) -> Result<Symbol, InternerExhausted> {
    if let Some(symbol) = symbol_of(id) {
        return Ok(symbol);
    }
    let table = table();
    let mut guard = table.write().expect("identity interner poisoned");
    // Re-check: another writer may have interned between the read and the
    // write lock.
    if let Some(symbol) = guard.symbols.get(id).copied() {
        return Ok(symbol);
    }
    let next = u32::try_from(guard.strings.len()).map_err(|_| InternerExhausted)?;
    let leaked: &'static str = Box::leak(id.to_string().into_boxed_str());
    let symbol = Symbol(next);
    guard.symbols.insert(leaked, symbol);
    guard.strings.push(leaked);
    guard.bytes = guard.bytes.saturating_add(leaked.len());
    Ok(symbol)
}

/// Resolve an identity that has already been interned, without minting one.
///
/// Every read path (record lookup, candidate rehydration, reachability seeds)
/// goes through here, so an unknown id costs a hash probe and nothing else.
#[must_use]
pub fn symbol_of(id: &str) -> Option<Symbol> {
    table()
        .read()
        .expect("identity interner poisoned")
        .symbols
        .get(id)
        .copied()
}

/// Observe the identity table. Intended for churn oracles and residency
/// receipts, not for control flow.
#[must_use]
pub fn interner_stats() -> InternerStats {
    let guard = table().read().expect("identity interner poisoned");
    InternerStats {
        symbols: guard.strings.len(),
        bytes: guard.bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests never read `interner_stats`: the table is process-global, so
    // any count assertion would race the other tests sharing this binary. The
    // growth oracle lives in `tests/identity_interner_churn.rs`, which runs
    // alone in its own process.

    #[test]
    fn interning_the_same_id_returns_the_same_symbol() {
        let first = intern("intern-test:same");
        let second = intern(&String::from("intern-test:same"));
        assert_eq!(first, second);
        assert_eq!(first.as_str(), "intern-test:same");
    }

    #[test]
    fn distinct_ids_get_distinct_symbols() {
        let left = intern("intern-test:left");
        let right = intern("intern-test:right");
        assert_ne!(left, right);
        assert_eq!(left.as_str(), "intern-test:left");
        assert_eq!(right.as_str(), "intern-test:right");
    }

    #[test]
    fn lookup_never_mints_a_symbol() {
        assert!(symbol_of("intern-test:never-admitted").is_none());
        assert!(symbol_of("intern-test:never-admitted").is_none());
        let admitted = intern("intern-test:admitted");
        assert_eq!(symbol_of("intern-test:admitted"), Some(admitted));
    }

    #[test]
    fn concurrent_first_admission_converges_on_one_symbol() {
        let threads = (0..16)
            .map(|_| std::thread::spawn(|| intern("intern-test:concurrent")))
            .collect::<Vec<_>>();
        let symbols = threads
            .into_iter()
            .map(|thread| thread.join().expect("interner worker"))
            .collect::<Vec<_>>();

        assert!(symbols.iter().all(|symbol| *symbol == symbols[0]));
        assert_eq!(symbols[0].as_str(), "intern-test:concurrent");
    }
}
