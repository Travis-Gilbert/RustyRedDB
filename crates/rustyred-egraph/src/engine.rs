use crate::{
    ContentHash, CostModel, Error, ExecutionReceipt, ExtractionReceipt, FunctionRow, FunctionRows,
    RulePack, SaturationReceipt, StopReason, Term, ENGINE_ID,
};
use egglog::ast::{Command, GenericRunConfig, RustSpan, Span};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

static NEXT_ENGINE: AtomicU64 = AtomicU64::new(1);

/// Opaque root bound to one evaluator; a handle from another engine is refused.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermId {
    engine: u64,
    slot: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Schedule {
    Run {
        ruleset: String,
        iterations: usize,
    },
    FixedPoint {
        rulesets: Vec<String>,
        max_iterations: usize,
    },
}

pub struct Engine {
    pub(crate) inner: egglog::EGraph,
    pack: RulePack,
    identity: u64,
    roots: Vec<(egglog::ArcSort, egglog::Value, String)>,
    rulesets: BTreeSet<String>,
    initialization: Option<ExecutionReceipt>,
    pristine: bool,
}

impl Engine {
    pub fn new(pack: RulePack) -> Result<Self, Error> {
        pack.validate()?;
        let commands = egglog::ast::Parser::default()
            .get_program_from_string(None, &pack.program)
            .map_err(|e| Error::Egglog(e.to_string()))?;
        for command in &commands {
            admit_pure(command)?;
        }
        Self::with_commands(pack, commands)
    }
    pub(crate) fn with_commands(pack: RulePack, commands: Vec<Command>) -> Result<Self, Error> {
        pack.validate()?;
        let mut engine = Self {
            inner: egglog::EGraph::default(),
            pack,
            identity: NEXT_ENGINE.fetch_add(1, Ordering::Relaxed),
            roots: Vec::new(),
            rulesets: BTreeSet::from([String::new()]),
            initialization: None,
            pristine: true,
        };
        engine.initialization = Some(engine.execute_commands(commands)?.0);
        engine.pristine = true;
        Ok(engine)
    }
    /// No caller commands, roots, or schedules have touched this initialized pack.
    pub fn is_pristine(&self) -> bool {
        self.pristine
    }
    pub fn rule_pack(&self) -> &RulePack {
        &self.pack
    }
    pub fn initialization(&self) -> &ExecutionReceipt {
        self.initialization.as_ref().expect("initialized engine")
    }
    pub fn num_tuples(&self) -> usize {
        self.inner.num_tuples()
    }
    pub fn has_function(&self, name: &str) -> bool {
        self.inner.get_function(name).is_some()
    }
    pub(crate) fn clone_baseline(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            pack: self.pack.clone(),
            identity: NEXT_ENGINE.fetch_add(1, Ordering::Relaxed),
            roots: Vec::new(),
            rulesets: self.rulesets.clone(),
            initialization: self.initialization.clone(),
            pristine: self.pristine,
        }
    }

    /// Execute caller-owned facts/checks using the same pinned evaluator. File,
    /// macro and external command execution is excluded from this pure crate.
    pub fn execute(&mut self, source: &str) -> Result<ExecutionReceipt, Error> {
        let commands = self
            .inner
            .parser
            .get_program_from_string(None, source)
            .map_err(|e| Error::Egglog(e.to_string()))?;
        for command in &commands {
            admit_pure(command)?;
        }
        Ok(self.execute_commands(commands)?.0)
    }
    pub(crate) fn execute_commands(
        &mut self,
        commands: Vec<Command>,
    ) -> Result<(ExecutionReceipt, bool), Error> {
        let start = Instant::now();
        let mut rulesets = self.rulesets.clone();
        for command in &commands {
            validate_schedule_names(command, &mut rulesets)?;
        }
        self.pristine = false;
        let outputs = self.inner.run_program(commands)?;
        self.rulesets = rulesets;
        let mut receipt = self.observation();
        let mut updated = false;
        for output in &outputs {
            if let egglog::CommandOutput::RunSchedule(report) = output {
                updated |= report.updated;
                receipt.iterations += report.iterations.len();
                for (rule, count) in &report.num_matches_per_rule {
                    *receipt
                        .matches_per_rule
                        .entry(rule.to_string())
                        .or_default() += count;
                }
            }
        }
        receipt.matches = receipt.matches_per_rule.values().sum();
        receipt.wall_time_ns = start.elapsed().as_nanos();
        Ok((
            ExecutionReceipt {
                output_count: outputs.len(),
                saturation: receipt,
            },
            updated,
        ))
    }
    fn observation(&self) -> SaturationReceipt {
        let graph = self.inner.serialize(egglog::SerializeConfig::default());
        debug_assert!(graph.is_complete());
        let classes = graph
            .egraph
            .nodes
            .values()
            .map(|node| &node.eclass)
            .collect::<std::collections::HashSet<_>>();
        SaturationReceipt {
            engine: ENGINE_ID.into(),
            rule_pack: self.pack.id.clone(),
            iterations: 0,
            matches: 0,
            matches_per_rule: BTreeMap::new(),
            eclasses: classes.len(),
            enodes: graph.egraph.nodes.len(),
            tuples: self.inner.num_tuples(),
            wall_time_ns: 0,
            stop_reason: StopReason::Scheduled,
        }
    }
    pub fn saturate(&mut self, schedule: Schedule) -> Result<SaturationReceipt, Error> {
        let start = Instant::now();
        let (rulesets, limit, fixed) = match schedule {
            Schedule::Run {
                ruleset,
                iterations,
            } => (vec![ruleset], iterations, false),
            Schedule::FixedPoint {
                rulesets,
                max_iterations,
            } => (rulesets, max_iterations, true),
        };
        if rulesets.is_empty() {
            return Err(Error::Invalid(
                "a schedule must name at least one ruleset".into(),
            ));
        }
        if let Some(name) = rulesets.iter().find(|name| !self.rulesets.contains(*name)) {
            return Err(Error::Invalid(format!("unknown ruleset {name}")));
        }
        self.pristine = false;
        let mut total = self.observation();
        total.stop_reason = if fixed {
            StopReason::IterationLimit
        } else {
            StopReason::Scheduled
        };
        while total.iterations < limit {
            let remaining = limit - total.iterations;
            let commands = rulesets
                .iter()
                .take(remaining)
                .map(|name| {
                    Command::RunSchedule(egglog::ast::Schedule::Run(
                        egglog::span!(),
                        GenericRunConfig {
                            ruleset: name.clone(),
                            until: None,
                        },
                    ))
                })
                .collect();
            let (step, updated) = self.execute_commands(commands)?;
            total.iterations += step.saturation.iterations;
            for (rule, count) in step.saturation.matches_per_rule {
                *total.matches_per_rule.entry(rule).or_default() += count;
            }
            total.eclasses = step.saturation.eclasses;
            total.enodes = step.saturation.enodes;
            total.tuples = step.saturation.tuples;
            if fixed && !updated && remaining >= rulesets.len() {
                total.stop_reason = StopReason::FixedPoint;
                break;
            }
        }
        total.matches = total.matches_per_rule.values().sum();
        total.wall_time_ns = start.elapsed().as_nanos();
        Ok(total)
    }
    pub fn root(&mut self, expression: &str) -> Result<TermId, Error> {
        self.pristine = false;
        let expr = self
            .inner
            .parser
            .get_expr_from_string(None, expression)
            .map_err(|e| Error::Egglog(e.to_string()))?;
        let (sort, value) = self.inner.eval_expr(&expr)?;
        let slot = self.roots.len();
        self.roots.push((sort, value, expression.into()));
        Ok(TermId {
            engine: self.identity,
            slot,
        })
    }
    pub fn extract(
        &self,
        root: TermId,
        cost: &CostModel,
    ) -> Result<(Term, ExtractionReceipt), Error> {
        if root.engine != self.identity {
            return Err(Error::Invalid("root belongs to a different engine".into()));
        }
        let (sort, value, source) = self
            .roots
            .get(root.slot)
            .ok_or_else(|| Error::Invalid("unknown root".into()))?;
        let start = Instant::now();
        let extractor = egglog::extract::Extractor::compute_costs_from_rootsorts(
            Some(vec![sort.clone()]),
            &self.inner,
            AdditiveCost(cost.clone()),
        );
        let mut dag = egglog::TermDag::default();
        // A saved root can have been merged since it was registered. Extraction
        // resolves the current representative exactly as an egglog expression
        // lookup does; retaining its old union-find value loses valid terms.
        let value = self.inner.get_canonical_value(*value, sort);
        let (weight, id) = extractor
            .extract_best_with_sort(&self.inner, &mut dag, value, sort.clone())
            .ok_or_else(|| Error::Invalid("root has no extractable term".into()))?;
        let term = crate::term::decode(&dag, id);
        let receipt = ExtractionReceipt {
            engine: ENGINE_ID.into(),
            rule_pack: self.pack.id.clone(),
            cost_model: cost.content_hash(),
            root: source.clone(),
            term: term.expression.clone(),
            term_hash: ContentHash::digest(term.expression.as_bytes()),
            cost: weight,
            wall_time_ns: start.elapsed().as_nanos(),
        };
        Ok((term, receipt))
    }
    pub fn function_rows(&self, name: &str) -> Result<FunctionRows, Error> {
        let function = self
            .inner
            .get_function(name)
            .ok_or_else(|| Error::Invalid(format!("unknown function {name}")))?;
        let (inputs, outputs, dag) = self.inner.function_to_dag(name, usize::MAX, true)?;
        let rows = inputs
            .into_iter()
            .zip(outputs.expect("requested output column"))
            .map(|(input, output)| {
                let egglog::Term::App(_, children) = dag.get(input) else {
                    unreachable!("egglog table keys are applications")
                };
                FunctionRow {
                    inputs: children
                        .iter()
                        .map(|id| crate::term::decode(&dag, *id))
                        .collect(),
                    output: crate::term::decode(&dag, output),
                }
            })
            .collect();
        Ok(FunctionRows {
            output_sort: function.schema().output.name().to_string(),
            rows,
        })
    }
}

fn admit_pure(command: &Command) -> Result<(), Error> {
    match command {
        Command::Push(..) | Command::Pop(..) => Err(Error::Invalid(
            "stack restoration would invalidate retained root handles".into(),
        )),
        Command::Input { .. }
        | Command::Output { .. }
        | Command::Include(..)
        | Command::UserDefined(..)
        | Command::PrintOverallStatistics(_, Some(_))
        | Command::PrintFunction(_, _, _, Some(_), _) => Err(Error::Invalid(
            "external I/O is not admitted by the pure engine".into(),
        )),
        Command::Fail(_, inner) => admit_pure(inner),
        _ => Ok(()),
    }
}

fn validate_schedule_names(command: &Command, known: &mut BTreeSet<String>) -> Result<(), Error> {
    match command {
        Command::AddRuleset(_, name) => {
            if !known.insert(name.clone()) {
                return Err(Error::Invalid(format!("duplicate ruleset {name}")));
            }
        }
        Command::UnstableCombinedRuleset(_, name, children) => {
            if known.contains(name) {
                return Err(Error::Invalid(format!("duplicate ruleset {name}")));
            }
            if children.iter().any(|child| !known.contains(child)) {
                return Err(Error::Invalid(
                    "combined ruleset contains an unknown ruleset".into(),
                ));
            }
            known.insert(name.clone());
        }
        Command::RunSchedule(schedule) => validate_schedule(schedule, known)?,
        Command::Fail(_, inner) => validate_schedule_names(inner, &mut known.clone())?,
        _ => {}
    }
    Ok(())
}
fn validate_schedule(
    schedule: &egglog::ast::Schedule,
    known: &BTreeSet<String>,
) -> Result<(), Error> {
    match schedule {
        egglog::ast::Schedule::Run(_, run) if !known.contains(&run.ruleset) => {
            Err(Error::Invalid(format!("unknown ruleset {}", run.ruleset)))
        }
        egglog::ast::Schedule::Run(..) => Ok(()),
        egglog::ast::Schedule::Repeat(_, _, inner) | egglog::ast::Schedule::Saturate(_, inner) => {
            validate_schedule(inner, known)
        }
        egglog::ast::Schedule::Sequence(_, children) => {
            for child in children {
                validate_schedule(child, known)?;
            }
            Ok(())
        }
    }
}
struct AdditiveCost(CostModel);
impl egglog::extract::CostModel<u64> for AdditiveCost {
    fn fold(&self, _head: &str, children: &[u64], head: u64) -> u64 {
        children
            .iter()
            .fold(head, |total, cost| total.saturating_add(*cost))
    }
    fn enode_cost(
        &self,
        graph: &egglog::EGraph,
        function: &egglog::Function,
        row: &egglog::FunctionRow,
    ) -> u64 {
        self.0
            .constructor_costs
            .get(function.name())
            .copied()
            .unwrap_or_else(|| {
                egglog::extract::TreeAdditiveCostModel::default().enode_cost(graph, function, row)
            })
    }
}
