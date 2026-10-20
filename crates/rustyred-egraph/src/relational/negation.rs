//! Stratification and compilation to the pinned evaluator's Set anti-join.
use super::*;
use egglog::ast::{GenericActions, GenericRule, Rule};
use egglog::{ArcSort, ExecutionState, Primitive, Value};
use std::time::Instant;

const SET_SORT: &str = "__theorem_relational_closed_set";

#[derive(Clone)]
pub(super) struct Negative {
    pub id: usize,
    pub relation: String,
    pub columns: Vec<usize>,
    pub arity: usize,
    pub sorts: Vec<String>,
    pub stratum: usize,
}
impl Negative {
    fn key(&self) -> String {
        format!("{RESERVED_PREFIX}key_{}", self.id)
    }
    fn table(&self) -> String {
        format!("{RESERVED_PREFIX}closed_{}", self.id)
    }
}

pub(super) struct NegativeWitness {
    pub id: usize,
    pub values: Vec<Expr>,
}
pub(super) struct Plan {
    pub negatives: Vec<Negative>,
    rule_levels: BTreeMap<usize, usize>,
    body_negatives: BTreeMap<(usize, usize), usize>,
    levels: BTreeSet<usize>,
}

fn stratum_name(level: usize) -> String {
    format!("{RESERVED_PREFIX}stratum_{level}")
}
fn freeze_name(level: usize) -> String {
    format!("{RESERVED_PREFIX}freeze_{level}")
}
fn call(name: impl Into<String>, args: Vec<Expr>) -> Expr {
    Expr::Call(egglog::span!(), name.into(), args)
}
fn var(name: impl Into<String>) -> Expr {
    Expr::Var(egglog::span!(), name.into())
}

fn negative(expr: &Expr, relations: &Relations) -> Option<(String, Vec<Expr>)> {
    let Expr::Call(_, name, args) = expr else {
        return None;
    };
    if name != "not" || args.len() != 1 {
        return None;
    }
    let Expr::Call(_, relation, columns) = &args[0] else {
        return None;
    };
    relations
        .contains_key(relation)
        .then(|| (relation.clone(), columns.clone()))
}
// The pinned parser replaces each source `_` with its own fresh variable.
// Its retained source span distinguishes anonymity without guessing generated names.
fn anonymous(expr: &Expr) -> bool {
    matches!(expr, Expr::Var(Span::Egglog(span), _) if &span.file.contents[span.i..span.j] == "_")
}
fn contains_anonymous(expr: &Expr) -> bool {
    anonymous(expr) || matches!(expr, Expr::Call(_, _, args) if args.iter().any(contains_anonymous))
}

fn vars(expr: &Expr, out: &mut BTreeSet<String>) {
    match expr {
        Expr::Var(_, name) => {
            out.insert(name.clone());
        }
        Expr::Call(_, _, args) => {
            for arg in args {
                vars(arg, out)
            }
        }
        _ => {}
    }
}
fn expr_vars(expr: &Expr) -> BTreeSet<String> {
    let mut set = BTreeSet::new();
    vars(expr, &mut set);
    set
}

impl Plan {
    pub fn new(commands: &[Command], relations: &Relations) -> Result<Self, Error> {
        let mut dependencies = Vec::<(Vec<String>, Vec<(String, bool)>)>::new();
        let mut by_rule = BTreeMap::new();
        let mut negatives = Vec::new();
        let mut body_negatives = BTreeMap::new();
        for (index, command) in commands.iter().enumerate() {
            let Command::Rule { rule } = command else {
                continue;
            };
            let mut heads = Vec::new();
            for action in &rule.head.0 {
                let Action::Expr(_, Expr::Call(_, name, args)) = action else {
                    return Err(refuse("a relational head may only insert tuples"));
                };
                validate_call(name, args, relations)?;
                heads.push(name.clone());
            }
            if heads.is_empty() {
                return Err(refuse("empty relational rule head"));
            }
            let mut deps = Vec::new();
            for (body_index, fact) in rule.body.iter().enumerate() {
                let Fact::Fact(expr) = fact else { continue };
                if let Some((name, args)) = negative(expr, relations) {
                    validate_call(&name, &args, relations)?;
                    let columns: Vec<_> = args
                        .iter()
                        .enumerate()
                        .filter_map(|(i, arg)| (!anonymous(arg)).then_some(i))
                        .collect();
                    let id = negatives.len();
                    negatives.push(Negative {
                        id,
                        arity: args.len(),
                        sorts: columns
                            .iter()
                            .map(|i| relations[&name][*i].clone())
                            .collect(),
                        columns,
                        relation: name.clone(),
                        stratum: 0,
                    });
                    body_negatives.insert((index, body_index), id);
                    deps.push((name, true));
                } else if let Expr::Call(_, name, _) = expr {
                    if relations.contains_key(name) {
                        deps.push((name.clone(), false));
                    }
                }
            }
            by_rule.insert(index, heads.clone());
            dependencies.push((heads, deps));
        }
        let mut relation_levels: BTreeMap<_, usize> =
            relations.keys().map(|name| (name.clone(), 0)).collect();
        for round in 0..=relations.len() {
            let mut changed = false;
            for (heads, deps) in &dependencies {
                let required = deps
                    .iter()
                    .map(|(name, negative)| relation_levels[name] + usize::from(*negative))
                    .chain(heads.iter().map(|name| relation_levels[name]))
                    .max()
                    .unwrap_or(0);
                for head in heads {
                    if relation_levels[head] < required {
                        relation_levels.insert(head.clone(), required);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
            if round == relations.len() {
                return Err(refuse("negative dependency cycle is not stratifiable"));
            }
        }
        let rule_levels: BTreeMap<_, _> = by_rule
            .into_iter()
            .map(|(index, heads)| {
                (
                    index,
                    heads
                        .iter()
                        .map(|name| relation_levels[name])
                        .max()
                        .unwrap_or(0),
                )
            })
            .collect();
        for ((rule, _), id) in &body_negatives {
            negatives[*id].stratum = rule_levels[rule];
        }
        let levels = rule_levels
            .values()
            .copied()
            .chain(std::iter::once(0))
            .collect();
        Ok(Self {
            negatives,
            rule_levels,
            body_negatives,
            levels,
        })
    }

    pub fn definitions(&self) -> Result<Vec<Command>, Error> {
        let mut source = String::new();
        for level in &self.levels {
            source.push_str(&format!("(ruleset {})\n", stratum_name(*level)));
        }
        if !self.negatives.is_empty() {
            source.push_str(&format!("(sort {SET_SORT} (Set String))\n"));
        }
        let freezes: BTreeSet<_> = self
            .negatives
            .iter()
            .map(|negative| negative.stratum)
            .collect();
        for level in freezes {
            source.push_str(&format!("(ruleset {})\n", freeze_name(level)));
        }
        for negative in &self.negatives {
            source.push_str(&format!(
                "(function {} () {SET_SORT} :merge (set-union old new))\n",
                negative.table()
            ));
        }
        egglog::ast::Parser::default()
            .get_program_from_string(None, &source)
            .map_err(|error| refuse(error.to_string()))
    }

    pub fn rewrite(
        &self,
        index: usize,
        rule: &mut Rule,
        relations: &Relations,
    ) -> Result<Vec<NegativeWitness>, Error> {
        // Every anonymous positive column is independently bound, and recorded
        // in the matched witness. It cannot become an unbound head wildcard.
        let mut all_variables = BTreeSet::new();
        for fact in &rule.body {
            match fact {
                Fact::Fact(expr) => vars(expr, &mut all_variables),
                Fact::Eq(_, a, b) => {
                    vars(a, &mut all_variables);
                    vars(b, &mut all_variables);
                }
            }
        }
        for action in &rule.head.0 {
            if let Action::Expr(_, expr) = action {
                vars(expr, &mut all_variables);
            }
        }
        if all_variables
            .iter()
            .any(|name| name.starts_with(RESERVED_PREFIX))
        {
            return Err(refuse("reserved relational variable prefix"));
        }
        let mut ordinal = 0;
        for fact in &mut rule.body {
            if let Fact::Fact(Expr::Call(_, name, args)) = fact {
                if relations.contains_key(name) {
                    for arg in args {
                        if anonymous(arg) {
                            *arg = var(format!("{RESERVED_PREFIX}wild_{index}_{ordinal}"));
                            ordinal += 1;
                        }
                    }
                }
            }
        }
        let mut bound = BTreeSet::new();
        for fact in &rule.body {
            if let Fact::Fact(Expr::Call(_, name, args)) = fact {
                if relations.contains_key(name) {
                    for arg in args {
                        if let Expr::Var(_, name) = arg {
                            bound.insert(name.clone());
                        }
                    }
                }
            }
        }
        // Scalar equalities may bind a name from already-grounded expressions.
        loop {
            let before = bound.len();
            for fact in &rule.body {
                if let Fact::Eq(_, left, right) = fact {
                    if let Expr::Var(_, name) = left {
                        if expr_vars(right).is_subset(&bound) {
                            bound.insert(name.clone());
                        }
                    }
                    if let Expr::Var(_, name) = right {
                        if expr_vars(left).is_subset(&bound) {
                            bound.insert(name.clone());
                        }
                    }
                }
            }
            if bound.len() == before {
                break;
            }
        }
        let check = |expr: &Expr| -> Result<(), Error> {
            let used = expr_vars(expr);
            if contains_anonymous(expr) || !used.is_subset(&bound) {
                return Err(refuse(format!(
                    "unsafe or anonymous variable outside a bound relational column: {expr}; bound={bound:?}",
                )));
            }
            validate_scalar(expr)
        };
        let mut witnesses = Vec::new();
        for (body_index, fact) in rule.body.iter_mut().enumerate() {
            if let Some(id) = self.body_negatives.get(&(index, body_index)) {
                let Fact::Fact(expr) = fact else {
                    unreachable!()
                };
                let (_, args) = negative(expr, relations).expect("validated negative body");
                let spec = &self.negatives[*id];
                let values: Vec<_> = spec.columns.iter().map(|i| args[*i].clone()).collect();
                for value in &values {
                    check(value)?;
                }
                *expr = call(
                    "set-not-contains",
                    vec![call(spec.table(), vec![]), call(spec.key(), values.clone())],
                );
                witnesses.push(NegativeWitness { id: *id, values });
            } else {
                match fact {
                    Fact::Fact(Expr::Call(_, name, args)) if relations.contains_key(name) => {
                        for arg in args {
                            check(arg)?
                        }
                    }
                    Fact::Fact(expr) => check(expr)?,
                    Fact::Eq(_, left, right) => {
                        check(left)?;
                        check(right)?;
                    }
                }
            }
        }
        for action in &rule.head.0 {
            if let Action::Expr(_, Expr::Call(_, _, args)) = action {
                for arg in args {
                    check(arg)?;
                }
            }
        }
        rule.ruleset = stratum_name(self.rule_levels[&index]);
        Ok(witnesses)
    }

    pub fn install(
        &self,
        engine: &mut Engine,
        relations: &Relations,
    ) -> Result<Vec<Command>, Error> {
        let mut rules = Vec::new();
        for negative in &self.negatives {
            let sorts = negative
                .sorts
                .iter()
                .map(|name| {
                    engine
                        .inner
                        .get_sort_by_name(name)
                        .cloned()
                        .expect("validated scalar sort")
                })
                .chain(std::iter::once(
                    engine.inner.get_sort_by_name("String").unwrap().clone(),
                ))
                .collect();
            engine.inner.add_primitive(TupleKey {
                name: negative.key(),
                sorts,
                kinds: negative.sorts.clone(),
            });
            let args: Vec<_> = (0..negative.arity)
                .map(|i| var(format!("{RESERVED_PREFIX}column_{i}")))
                .collect();
            let key_args = negative.columns.iter().map(|i| args[*i].clone()).collect();
            rules.push(Command::Rule {
                rule: GenericRule {
                    span: egglog::span!(),
                    name: format!("{RESERVED_PREFIX}seal_{}", negative.id),
                    ruleset: freeze_name(negative.stratum),
                    body: vec![Fact::Fact(call(&negative.relation, args))],
                    head: GenericActions(vec![Action::Set(
                        egglog::span!(),
                        negative.table(),
                        vec![],
                        call("set-of", vec![call(negative.key(), key_args)]),
                    )]),
                },
            });
            debug_assert_eq!(negative.arity, relations[&negative.relation].len());
        }
        Ok(rules)
    }

    pub fn run(
        &self,
        engine: &mut Engine,
        limit: usize,
    ) -> Result<(SaturationReceipt, BTreeMap<usize, ContentHash>), Error> {
        let start = Instant::now();
        let mut total = engine.saturate(Schedule::Run {
            ruleset: stratum_name(0),
            iterations: 0,
        })?;
        total.stop_reason = crate::StopReason::IterationLimit;
        let mut hashes = BTreeMap::new();
        for level in &self.levels {
            let negatives: Vec<_> = self
                .negatives
                .iter()
                .filter(|negative| negative.stratum == *level)
                .collect();
            if !negatives.is_empty() {
                if total.iterations >= limit {
                    break;
                }
                let mut init = Vec::new();
                for negative in negatives {
                    let rows = read_rows(engine, &negative.relation)?;
                    hashes.insert(
                        negative.id,
                        ContentHash::digest(canonical(&(&negative.relation, rows))),
                    );
                    init.push(Command::Action(Action::Set(
                        egglog::span!(),
                        negative.table(),
                        vec![],
                        call("set-empty", vec![]),
                    )));
                }
                engine.execute_commands(init)?;
                let step = engine.saturate(Schedule::FixedPoint {
                    rulesets: vec![freeze_name(*level)],
                    max_iterations: limit - total.iterations,
                })?;
                let complete = step.stop_reason == crate::StopReason::FixedPoint;
                accumulate(&mut total, step);
                if !complete {
                    break;
                }
            }
            if total.iterations >= limit {
                break;
            }
            let step = engine.saturate(Schedule::FixedPoint {
                rulesets: vec![stratum_name(*level)],
                max_iterations: limit - total.iterations,
            })?;
            let complete = step.stop_reason == crate::StopReason::FixedPoint;
            accumulate(&mut total, step);
            if !complete {
                break;
            }
            if Some(level) == self.levels.last() {
                total.stop_reason = crate::StopReason::FixedPoint;
            }
        }
        total.wall_time_ns = start.elapsed().as_nanos();
        Ok((total, hashes))
    }
}

fn accumulate(total: &mut SaturationReceipt, step: SaturationReceipt) {
    total.iterations += step.iterations;
    for (rule, count) in step.matches_per_rule {
        *total.matches_per_rule.entry(rule).or_default() += count;
    }
    total.matches = total.matches_per_rule.values().sum();
    total.eclasses = step.eclasses;
    total.enodes = step.enodes;
    total.tuples = step.tuples;
}

/// Encoding only. Projection and absence testing are actual egglog Set rules
/// and set-not-contains calls, never a second Rust relation evaluator.
#[derive(Clone)]
struct TupleKey {
    name: String,
    sorts: Vec<ArcSort>,
    kinds: Vec<String>,
}
impl Primitive for TupleKey {
    fn name(&self) -> &str {
        &self.name
    }
    fn get_type_constraints(&self, span: &Span) -> Box<dyn egglog::constraint::TypeConstraint> {
        egglog::constraint::SimpleTypeConstraint::new(&self.name, self.sorts.clone(), span.clone())
            .into_box()
    }
    fn apply(&self, state: &mut ExecutionState, args: &[Value]) -> Option<Value> {
        let mut atoms = Vec::with_capacity(args.len());
        for (value, kind) in args.iter().zip(&self.kinds) {
            atoms.push(match kind.as_str() {
                "String" => Atom::String(state.base_values().unwrap::<egglog::sort::S>(*value).0),
                "i64" => Atom::I64(state.base_values().unwrap::<i64>(*value)),
                "bool" => Atom::Bool(state.base_values().unwrap::<bool>(*value)),
                "f64" => {
                    let number = **state.base_values().unwrap::<egglog::sort::F>(*value);
                    if !number.is_finite() {
                        return None;
                    }
                    Atom::F64(if number == 0.0 { 0.0 } else { number })
                }
                _ => return None,
            });
        }
        let key = serde_json::to_string(&atoms).ok()?;
        Some(state.base_values().get::<egglog::sort::S>(key.into()))
    }
}
