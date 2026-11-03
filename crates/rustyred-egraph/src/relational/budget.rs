//! The work budget: how much one derivation may join, record and build.
//!
//! A unit of work is one recorded value, or 64 bytes of a recorded or built
//! string. Every relational rule body starts with two charges:
//!
//! - [`BUDGET`] carries the rule's fixed cost: one for the binding, one for
//!   each value its witness rows record (the head tuple and every matched
//!   premise tuple), and the bytes of the rule's string literals and of the
//!   rule and relation names its provenance records.
//! - [`BYTES`] takes the binding's recorded string variables and charges their
//!   bytes.
//!
//! The pinned evaluator calls a body primitive once for each binding of the
//! rule's relation atoms, in body order, so the charges cover every candidate
//! match, including those the rule's scalar conditions then reject. The
//! compiler also renames every `+` to [`PLUS`], whose string overload charges
//! a concatenation's bytes before it allocates them; its `i64` and `f64`
//! overloads are egglog's own. Once a charge passes the limit the primitive
//! raises egglog's early stop: the join halts at its next check, and the
//! derivation is refused with `Error::WorkBudget` before any relation is read
//! back. The budget therefore bounds the join, the strings the evaluator
//! builds, and what reading the result back allocates.

use egglog::ast::Expr;
use egglog::sort::{F, S};
use egglog::{add_primitive, EGraph, ExecutionState};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The reserved names, so no declared relation or user call can take them.
pub(super) const BUDGET: &str = "__theorem_relational_budget";
pub(super) const BYTES: &str = "__theorem_relational_bytes";
pub(super) const PLUS: &str = "__theorem_relational_plus";

/// Bytes of string one unit of work covers.
pub(super) const UNIT_BYTES: usize = 64;

/// Work charged since the last reset, against one derivation's limit.
#[derive(Debug, Default)]
pub(super) struct Meter {
    used: AtomicU64,
    limit: AtomicU64,
}
impl Meter {
    pub(super) fn reset(&self, limit: u64) {
        self.used.store(0, Ordering::Relaxed);
        self.limit.store(limit, Ordering::Relaxed);
    }
    pub(super) fn exhausted(&self) -> bool {
        self.used.load(Ordering::Relaxed) > self.limit.load(Ordering::Relaxed)
    }
    /// Charge `cost`; past the limit, stop the evaluator and fail the call.
    fn charge(&self, state: &ExecutionState, cost: u64) -> Option<()> {
        let used = self
            .used
            .fetch_add(cost, Ordering::Relaxed)
            .saturating_add(cost);
        if used > self.limit.load(Ordering::Relaxed) {
            state.trigger_early_stop();
            return None;
        }
        Some(())
    }
}

/// Work beyond its value that a string of `len` bytes costs.
pub(super) fn blocks(len: usize) -> usize {
    len / UNIT_BYTES
}

/// Register the charges and the metered `+`. A clone of the egraph keeps the
/// same meter.
#[rustfmt::skip]
pub(super) fn install(egraph: &mut EGraph, meter: &Arc<Meter>) {
    add_primitive!(egraph, "__theorem_relational_budget" = {Arc::clone(meter): Arc<Meter>} |cost: i64| -?> () {
        self.ctx.charge(exec_state, cost.unsigned_abs())
    });
    add_primitive!(egraph, "__theorem_relational_bytes" = {Arc::clone(meter): Arc<Meter>} [values: S] -?> () {{
        // One value at a time, so measuring holds one copy at most.
        let cost = values.map(|value| blocks(value.len()) as u64).sum();
        self.ctx.charge(exec_state, cost)
    }});
    add_primitive!(egraph, "__theorem_relational_plus" = |a: i64, b: i64| -?> i64 { a.checked_add(b) });
    add_primitive!(egraph, "__theorem_relational_plus" = |a: F, b: F| -> F { a + b });
    add_primitive!(egraph, "__theorem_relational_plus" = {Arc::clone(meter): Arc<Meter>} [parts: S] -?> S {{
        let len: usize = parts.clone().map(|part| part.len()).sum();
        self.ctx.charge(exec_state, blocks(len) as u64)?;
        let mut joined = String::with_capacity(len);
        parts.for_each(|part| joined.push_str(part.as_str()));
        Some(joined.into())
    }});
}

/// Route every `+` in `expr` through [`PLUS`].
pub(super) fn meter_plus(expr: &mut Expr) {
    if let Expr::Call(_, name, args) = expr {
        if name == "+" {
            *name = PLUS.into();
        }
        args.iter_mut().for_each(meter_plus);
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::{BUDGET, BYTES, PLUS};
    use crate::StopReason;

    /// Each binding costs 7: one, plus `out`'s three values and three `a`s.
    const CROSS: &str = r#"
(relation a (i64))
(relation out (i64 i64 i64))
(rule ((a x) (a y) (a z)) ((out x y z)) :name "cross")
"#;

    fn engine(program: &str) -> RelationalEngine {
        RelationalEngine::new(RulePack::new(program, Default::default())).unwrap()
    }
    fn numbers(relation: &str, values: impl IntoIterator<Item = i64>) -> (String, Vec<Vec<Atom>>) {
        (
            relation.into(),
            values.into_iter().map(|n| vec![Atom::I64(n)]).collect(),
        )
    }
    fn strings(
        relation: &str,
        values: impl IntoIterator<Item = String>,
    ) -> (String, Vec<Vec<Atom>>) {
        (
            relation.into(),
            values.into_iter().map(|s| vec![Atom::String(s)]).collect(),
        )
    }
    fn spent(result: Result<RelationalReceipt, Error>, budget: u64) {
        match result {
            Err(Error::WorkBudget(limit)) => assert_eq!(limit, budget),
            other => panic!("expected a spent budget of {budget}: {other:?}"),
        }
    }

    #[test]
    fn a_cross_product_stops_at_its_work_budget() {
        let cross = engine(CROSS);
        // 10^9 bindings: the evaluator halts after about ten thousand.
        let large = BTreeMap::from([numbers("a", 0..1_000)]);
        let started = std::time::Instant::now();
        spent(cross.derive_within(&large, 64, 70_000), 70_000);
        assert!(started.elapsed().as_secs() < 30, "{:?}", started.elapsed());

        // The positive control on the same engine: 10^3 bindings fit, and the
        // meter starts again for each derive.
        let small = BTreeMap::from([numbers("a", 0..10)]);
        let complete = cross.derive_within(&small, 64, 70_000).unwrap();
        assert_eq!(complete.saturation.stop_reason, StopReason::FixedPoint);
        assert_eq!(complete.relations["out"].len(), 1_000);
        assert_eq!(complete.provenance.len(), 1_000);
        assert_eq!(
            cross.derive(&small, 64).unwrap().relations,
            complete.relations
        );
        // Exactly at the budget is within it.
        let exact = cross.derive_within(&small, 64, 7_000).unwrap();
        assert_eq!(exact.saturation.stop_reason, StopReason::FixedPoint);
        spent(cross.derive_within(&small, 64, 6_999), 6_999);
    }

    #[test]
    fn bindings_the_conditions_reject_still_spend_the_budget() {
        let rejected = engine(
            r#"(relation a (i64))
(relation out (i64 i64 i64))
(rule ((a x) (a y) (a z) (< (+ x (+ y z)) -1)) ((out x y z)) :name "never")"#,
        );
        // 2.7 * 10^7 bindings, none of which passes the condition.
        let large = BTreeMap::from([numbers("a", 0..300)]);
        spent(rejected.derive_within(&large, 64, 70_000), 70_000);

        let small = BTreeMap::from([numbers("a", 0..10)]);
        let complete = rejected.derive_within(&small, 64, 7_000).unwrap();
        assert_eq!(complete.saturation.stop_reason, StopReason::FixedPoint);
        assert!(complete.relations["out"].is_empty());
    }

    #[test]
    fn a_wide_rule_costs_what_it_records() {
        // One binding per `a` row, and each records 1 + 40 values.
        let premises = "(a x) ".repeat(40);
        let wide = engine(&format!(
            "(relation a (i64)) (relation b (i64)) (rule ({premises}) ((b x)) :name \"wide\")"
        ));
        let facts = BTreeMap::from([numbers("a", 0..100)]);
        let complete = wide.derive_within(&facts, 64, 100 * 42).unwrap();
        assert_eq!(complete.relations["b"].len(), 100);
        assert_eq!(complete.provenance[0].premises.len(), 40);
        spent(wide.derive_within(&facts, 64, 100 * 42 - 1), 100 * 42 - 1);
    }

    #[test]
    fn a_recorded_string_costs_one_more_for_every_64_bytes() {
        let cross = engine(
            r#"(relation a (String))
(relation out (String String String))
(rule ((a x) (a y) (a z)) ((out x y z)) :name "cross")"#,
        );
        // Six recorded strings of 64 bytes: 7 + 6 for each of 10^3 bindings.
        let long = BTreeMap::from([strings("a", (0..10).map(|n| format!("{n:064}")))]);
        let complete = cross.derive_within(&long, 64, 13_000).unwrap();
        assert_eq!(complete.relations["out"].len(), 1_000);
        spent(cross.derive_within(&long, 64, 12_999), 12_999);
        // The control: 63 bytes are within the value's own unit.
        let short = BTreeMap::from([strings("a", (0..10).map(|n| format!("{n:063}")))]);
        cross.derive_within(&short, 64, 7_000).unwrap();
    }

    #[test]
    fn literals_and_recorded_names_are_part_of_the_fixed_cost() {
        // 1 + `b`'s two values and `a`'s one, + 2 for the 128-byte literal and
        // 1 for the 64-byte rule name: 7 for each of 10 bindings.
        let literal = "l".repeat(128);
        let name = "n".repeat(64);
        let tagged = engine(&format!(
            r#"(relation a (i64)) (relation b (i64 String))
(rule ((a x)) ((b x "{literal}")) :name "{name}")"#
        ));
        let facts = BTreeMap::from([numbers("a", 0..10)]);
        let complete = tagged.derive_within(&facts, 64, 70).unwrap();
        assert_eq!(complete.relations["b"].len(), 10);
        spent(tagged.derive_within(&facts, 64, 69), 69);
    }

    #[test]
    fn a_concatenation_is_charged_before_it_is_built() {
        let grow = engine(
            r#"(relation s (String))
(rule ((s y)) ((s (+ y y))) :name "grow")"#,
        );
        // Doubling from 2 bytes passes 10^4 units (640 KB) before 20 rounds.
        let seed = BTreeMap::from([strings("s", ["ab".to_string()])]);
        spent(grow.derive_within(&seed, 20, 10_000), 10_000);
        // The control: five rounds build what string `+` builds.
        let complete = grow.derive_within(&seed, 5, 10_000).unwrap();
        let built = &complete.relations["s"];
        assert_eq!(built.len(), 6);
        for n in 0..6 {
            let doubled = vec![Atom::String("ab".repeat(1 << n))];
            assert!(built.contains(&doubled), "{n}");
        }

        // One action asking for 16 MB from a 16 KB value is refused, though
        // the value itself fits.
        let parts = " y".repeat(1_000);
        let burst = engine(&format!(
            r#"(relation s (String)) (relation t (String))
(rule ((s y)) ((t (+{parts}))) :name "burst")"#
        ));
        let seed = BTreeMap::from([strings("s", ["x".repeat(16 << 10)])]);
        spent(burst.derive_within(&seed, 1, 100_000), 100_000);
    }

    #[test]
    fn numeric_addition_keeps_egglog_semantics() {
        let sums = engine(
            r#"(relation a (i64)) (relation f (f64))
(relation next (i64)) (relation half (f64))
(rule ((a x) (= y (+ x 1))) ((next y)) :name "next")
(rule ((f x)) ((half (+ x 0.5))) :name "half")"#,
        );
        let facts = BTreeMap::from([
            numbers("a", [1, i64::MAX]),
            ("f".into(), vec![vec![Atom::F64(1.0)]]),
        ]);
        let derived = sums.derive(&facts, 64).unwrap().relations;
        // An overflowing sum fails its binding, as egglog's checked `+` does.
        assert_eq!(derived["next"], vec![vec![Atom::I64(2)]]);
        assert_eq!(derived["half"], vec![vec![Atom::F64(1.5)]]);
    }

    #[test]
    fn one_budget_covers_every_rule_and_stratum() {
        let stratified = engine(
            r#"(relation a (i64))
(relation b (i64))
(relation pair (i64 i64))
(relation unpaired (i64))
(rule ((a x) (a y)) ((pair x y)) :name "pair")
(rule ((b x) (not (pair x x))) ((unpaired x)) :name "unpaired")"#,
        );
        // 100 `pair` bindings at 5 each, then 11 `unpaired` bindings at 5
        // each: one, `unpaired`, `b`, and the two columns of the absent pair.
        let facts = BTreeMap::from([numbers("a", 0..10), numbers("b", -1..10)]);
        let complete = stratified.derive_within(&facts, 64, 555).unwrap();
        assert_eq!(complete.saturation.stop_reason, StopReason::FixedPoint);
        assert_eq!(complete.relations["unpaired"], vec![vec![Atom::I64(-1)]]);
        spent(stratified.derive_within(&facts, 64, 554), 554);
        // Enough for the lower stratum alone.
        spent(stratified.derive_within(&facts, 64, 500), 500);
    }

    #[test]
    fn concurrent_derives_on_one_engine_keep_their_own_budgets() {
        let cross = engine(CROSS);
        let large = BTreeMap::from([numbers("a", 0..1_000)]);
        let small = BTreeMap::from([numbers("a", 0..10)]);
        std::thread::scope(|scope| {
            let stopped = scope.spawn(|| cross.derive_within(&large, 64, 70_000));
            let complete: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| cross.derive_within(&small, 64, 7_000).unwrap()))
                .collect();
            spent(stopped.join().unwrap(), 70_000);
            for receipt in complete {
                let receipt = receipt.join().unwrap();
                assert_eq!(receipt.saturation.stop_reason, StopReason::FixedPoint);
                assert_eq!(receipt.relations["out"].len(), 1_000);
            }
        });
    }

    #[test]
    fn a_program_cannot_name_or_call_the_budget() {
        let refused = |program: &str| {
            RelationalEngine::new(RulePack::new(program, Default::default()))
                .err()
                .unwrap_or_else(|| panic!("admitted: {program}"))
                .to_string()
        };
        for name in [BUDGET, BYTES, PLUS] {
            assert!(refused(&format!("(relation {name} (i64))")).contains("reserved"));
            let call = format!(
                "(relation a (i64)) (relation b (i64)) (rule ((a x) ({name} 0)) ((b x)) :name \"r\")"
            );
            assert!(refused(&call).contains("unadmitted primitive"), "{name}");
        }
        // The same rule without the call is admitted.
        engine("(relation a (i64)) (relation b (i64)) (rule ((a x)) ((b x)) :name \"r\")");
    }
}
