use crate::{Atom, ContentHash, Engine, Error, RulePack, SaturationReceipt, Schedule, TermValue};
use egglog::ast::{Action, Command, Expr, Fact, Literal, RustSpan, Span};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const RESERVED_PREFIX: &str = "__theorem_relational_";
const WITNESS_PREFIX: &str = "__theorem_relational_witness_";
mod negation;
type Relations = BTreeMap<String, Vec<String>>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationalFact {
    pub relation: String,
    pub tuple: Vec<Atom>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationalAbsence {
    pub relation: String,
    /// None is an anonymous existential column in the closed relation.
    pub pattern: Vec<Option<Atom>>,
    pub closed_relation: ContentHash,
    pub input_facts: ContentHash,
    pub rule_pack: ContentHash,
    pub stratum: usize,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationalDerivation {
    pub rule: String,
    pub output: RelationalFact,
    pub premises: Vec<RelationalFact>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub negative_premises: Vec<RelationalAbsence>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationalReceipt {
    pub rule_pack: ContentHash,
    /// Ground literal seeds read from the validated program AST. Their authority
    /// is the rule pack hash, independently of the caller-supplied fact pack.
    pub program_facts: Vec<RelationalFact>,
    pub relations: BTreeMap<String, Vec<Vec<Atom>>>,
    pub provenance: Vec<RelationalDerivation>,
    /// Includes evaluator witness tables; these are actual measured counts.
    pub saturation: SaturationReceipt,
}
struct Witness {
    name: String,
    rule: String,
    output: (String, usize),
    premises: Vec<(String, usize)>,
    negatives: Vec<usize>,
}

/// Scalar Datalog on the same evaluator, with no user-defined equality sorts,
/// union, rewrites, mutation, external I/O, or opaque command execution.
pub struct RelationalEngine {
    engine: Engine,
    relations: Relations,
    negation: negation::Plan,
    witnesses: Vec<Witness>,
    program_facts: Vec<RelationalFact>,
}
impl RelationalEngine {
    pub fn new(pack: RulePack) -> Result<Self, Error> {
        pack.validate()?;
        let commands = egglog::ast::Parser::default()
            .get_program_from_string(None, &pack.program)
            .map_err(|e| Error::Relational(e.to_string()))?;
        let mut relations = Relations::new();
        let mut rulesets = BTreeSet::from([String::new()]);
        let mut rule_names = BTreeSet::new();
        for command in &commands {
            match command {
                Command::Relation { name, inputs, .. } => {
                    if name.starts_with(RESERVED_PREFIX) || relations.contains_key(name) {
                        return Err(refuse("duplicate or reserved relation name"));
                    }
                    for sort in inputs {
                        if !matches!(sort.as_str(), "String" | "i64" | "f64" | "bool") {
                            return Err(refuse(format!("non-scalar relation sort {sort}")));
                        }
                    }
                    relations.insert(name.clone(), inputs.clone());
                }
                Command::AddRuleset(_, name) => {
                    if name.starts_with(RESERVED_PREFIX) {
                        return Err(refuse("reserved ruleset name"));
                    }
                    rulesets.insert(name.clone());
                }
                Command::Rule { rule } => {
                    let name = if rule.name.is_empty() {
                        egglog::ast::desugar::rule_name(command)
                    } else {
                        rule.name.clone()
                    };
                    if name.starts_with(RESERVED_PREFIX) || !rule_names.insert(name) {
                        return Err(refuse("duplicate or reserved relational rule name"));
                    }
                }
                Command::Action(..) => {}
                // This whitelist is structural. Union hidden inside Fail,
                // a macro, or a rewrite never reaches the evaluator.
                _ => {
                    return Err(refuse(format!(
                    "command is not a scalar relation declaration, insertion, or rule: {command}"
                )))
                }
            }
        }
        let negation = negation::Plan::new(&commands, &relations)?;
        let mut definitions = Vec::new();
        let mut executable = Vec::new();
        let mut witnesses = Vec::new();
        let mut program_facts = Vec::new();
        for (index, command) in commands.into_iter().enumerate() {
            match command {
                Command::Rule { mut rule } => {
                    if rule.name.is_empty() {
                        rule.name =
                            egglog::ast::desugar::rule_name(&Command::Rule { rule: rule.clone() });
                    }
                    if !rulesets.contains(&rule.ruleset) {
                        return Err(refuse("rule names an undeclared ruleset"));
                    }
                    let negative_witnesses = negation.rewrite(index, &mut rule, &relations)?;
                    let mut premises = Vec::new();
                    for fact in &rule.body {
                        match fact {
                            Fact::Fact(Expr::Call(_, name, args))
                                if relations.contains_key(name) =>
                            {
                                validate_call(name, args, &relations)?;
                                premises.push((name.clone(), args.clone()));
                            }
                            // Only the checked compiler can introduce this name;
                            // user scalar calls were validated before rewriting.
                            Fact::Fact(Expr::Call(_, name, _)) if name == "set-not-contains" => {}
                            Fact::Fact(expr) => validate_scalar(expr)?,
                            Fact::Eq(_, left, right) => {
                                validate_scalar(left)?;
                                validate_scalar(right)?;
                            }
                        }
                    }
                    let mut additions = Vec::new();
                    for (head_index, action) in rule.head.0.iter().enumerate() {
                        let Action::Expr(_, Expr::Call(_, output, args)) = action else {
                            return Err(refuse(
                                "a relational rule head may only insert relation tuples",
                            ));
                        };
                        validate_call(output, args, &relations)?;
                        let name = format!("{WITNESS_PREFIX}{index}_{head_index}");
                        let mut inputs = relations[output].clone();
                        let mut values = args.clone();
                        let mut proof_premises = Vec::new();
                        for (relation, args) in &premises {
                            inputs.extend(relations[relation].clone());
                            values.extend(args.clone());
                            proof_premises.push((relation.clone(), args.len()));
                        }
                        let mut negatives = Vec::new();
                        for negative in &negative_witnesses {
                            inputs.extend(negation.negatives[negative.id].sorts.clone());
                            values.extend(negative.values.clone());
                            negatives.push(negative.id);
                        }
                        definitions.push(Command::Relation {
                            span: egglog::span!(),
                            name: name.clone(),
                            inputs,
                        });
                        additions.push(Action::Expr(
                            egglog::span!(),
                            Expr::Call(egglog::span!(), name.clone(), values),
                        ));
                        witnesses.push(Witness {
                            name,
                            rule: if rule.name.is_empty() {
                                format!("rule-{index}")
                            } else {
                                rule.name.clone()
                            },
                            output: (output.clone(), args.len()),
                            premises: proof_premises,
                            negatives,
                        });
                    }
                    if rule.head.0.is_empty() {
                        return Err(refuse("empty relational rule head"));
                    }
                    rule.head.0.extend(additions);
                    executable.push(Command::Rule { rule });
                }
                Command::Action(action) => {
                    let Action::Expr(_, Expr::Call(_, name, args)) = &action else {
                        return Err(refuse(
                            "top-level actions may only insert literal relation tuples",
                        ));
                    };
                    validate_call(name, args, &relations)?;
                    if args.iter().any(|arg| !matches!(arg, Expr::Lit(..))) {
                        return Err(refuse("program facts must contain literal scalar values"));
                    }
                    let tuple = args
                        .iter()
                        .map(|arg| match arg {
                            Expr::Lit(_, Literal::String(value)) => Atom::String(value.clone()),
                            Expr::Lit(_, Literal::Int(value)) => Atom::I64(*value),
                            Expr::Lit(_, Literal::Float(value)) => Atom::F64(**value),
                            Expr::Lit(_, Literal::Bool(value)) => Atom::Bool(*value),
                            _ => unreachable!("the scalar literal gate precedes seed projection"),
                        })
                        .collect();
                    program_facts.push(RelationalFact {
                        relation: name.clone(),
                        tuple,
                    });
                    executable.push(Command::Action(action));
                }
                command => definitions.push(command),
            }
        }
        definitions.extend(negation.definitions()?);
        let mut engine = Engine::with_commands(pack, definitions)?;
        executable.extend(negation.install(&mut engine, &relations)?);
        engine.execute_commands(executable)?;
        program_facts.sort_by_cached_key(canonical);
        program_facts.dedup();
        Ok(Self {
            engine,
            relations,
            negation,
            witnesses,
            program_facts,
        })
    }

    /// Run one independent fact pack. Reusing this value does not retain a prior
    /// invocation's facts: cloning the validated baseline preserves isolation.
    pub fn derive(
        &self,
        facts: &BTreeMap<String, Vec<Vec<Atom>>>,
        max_iterations: usize,
    ) -> Result<RelationalReceipt, Error> {
        // Validate the entire pack before executing even its first insertion.
        let mut insertions = Vec::new();
        for (name, rows) in facts {
            let sorts = self
                .relations
                .get(name)
                .ok_or_else(|| refuse(format!("facts name undeclared relation {name}")))?;
            for row in rows {
                if row.len() != sorts.len() {
                    return Err(refuse(format!("wrong arity for {name}")));
                }
                let mut args = Vec::new();
                for (value, sort) in row.iter().zip(sorts) {
                    let matches = matches!(
                        (value, sort.as_str()),
                        (Atom::String(_), "String")
                            | (Atom::I64(_), "i64")
                            | (Atom::F64(_), "f64")
                            | (Atom::Bool(_), "bool")
                    );
                    if !matches {
                        return Err(refuse(format!("wrong scalar type for {name}")));
                    }
                    args.push(Expr::Lit(egglog::span!(), value.literal()?));
                }
                insertions.push(Command::Action(Action::Expr(
                    egglog::span!(),
                    Expr::Call(egglog::span!(), name.clone(), args),
                )));
            }
        }
        let mut engine = self.engine.clone_baseline();
        engine.execute_commands(insertions)?;
        let (saturation, closed_hashes) = self.negation.run(&mut engine, max_iterations)?;
        let mut canonical_input = facts.clone();
        for rows in canonical_input.values_mut() {
            rows.sort_by_cached_key(canonical);
            rows.dedup();
        }
        let input_facts = ContentHash::digest(canonical(&canonical_input));
        let mut relations = BTreeMap::new();
        for name in self.relations.keys() {
            relations.insert(name.clone(), read_rows(&engine, name)?);
        }
        let mut provenance = Vec::new();
        for witness in &self.witnesses {
            for tuple in read_rows(&engine, &witness.name)? {
                let mut offset = witness.output.1;
                let output = RelationalFact {
                    relation: witness.output.0.clone(),
                    tuple: tuple[..offset].to_vec(),
                };
                let mut premises = Vec::new();
                for (name, arity) in &witness.premises {
                    premises.push(RelationalFact {
                        relation: name.clone(),
                        tuple: tuple[offset..offset + arity].to_vec(),
                    });
                    offset += arity;
                }
                let mut negative_premises = Vec::new();
                for id in &witness.negatives {
                    let spec = &self.negation.negatives[*id];
                    let mut pattern = vec![None; spec.arity];
                    for column in &spec.columns {
                        pattern[*column] = Some(tuple[offset].clone());
                        offset += 1;
                    }
                    negative_premises.push(RelationalAbsence {
                        relation: spec.relation.clone(),
                        pattern,
                        closed_relation: closed_hashes
                            .get(id)
                            .ok_or_else(|| refuse("negative witness has no sealed lower relation"))?
                            .clone(),
                        input_facts: input_facts.clone(),
                        rule_pack: self.engine.rule_pack().id.clone(),
                        stratum: spec.stratum,
                    });
                }
                provenance.push(RelationalDerivation {
                    rule: witness.rule.clone(),
                    output,
                    premises,
                    negative_premises,
                });
            }
        }
        provenance.sort_by_cached_key(canonical);
        provenance.dedup();
        Ok(RelationalReceipt {
            rule_pack: self.engine.rule_pack().id.clone(),
            program_facts: self.program_facts.clone(),
            relations,
            provenance,
            saturation,
        })
    }
}
fn canonical(value: &impl Serialize) -> Vec<u8> {
    serde_json::to_vec(value).expect("validated finite scalar serialization")
}
fn read_rows(engine: &Engine, name: &str) -> Result<Vec<Vec<Atom>>, Error> {
    let mut rows = Vec::new();
    for row in engine.function_rows(name)?.rows {
        let mut values = Vec::new();
        for term in row.inputs {
            let TermValue::Atom(atom) = term.value else {
                return Err(refuse("evaluator returned a non-scalar relational row"));
            };
            atom.literal()?;
            values.push(atom);
        }
        rows.push(values);
    }
    rows.sort_by_cached_key(canonical);
    rows.dedup();
    Ok(rows)
}
fn validate_call(name: &str, args: &[Expr], relations: &Relations) -> Result<(), Error> {
    let sorts = relations
        .get(name)
        .ok_or_else(|| refuse(format!("head does not name a declared relation: {name}")))?;
    if sorts.len() != args.len() {
        return Err(refuse(format!("wrong relation arity: {name}")));
    }
    for arg in args {
        validate_scalar(arg)?;
    }
    Ok(())
}
fn validate_scalar(expr: &Expr) -> Result<(), Error> {
    match expr {
        Expr::Var(..) => Ok(()),
        Expr::Lit(_, Literal::Unit) => Err(refuse("Unit is not an admitted scalar column")),
        Expr::Lit(_, Literal::Float(x)) if !x.is_finite() => {
            Err(refuse("non-finite program float"))
        }
        Expr::Lit(..) => Ok(()),
        Expr::Call(_, name, args) => {
            // Deterministic scalar primitives only, never relation/function calls
            // hidden in scalar position or equality-producing constructs.
            if !matches!(
                name.as_str(),
                "+" | "-"
                    | "*"
                    | "/"
                    | "%"
                    | "min"
                    | "max"
                    | "abs"
                    | "<"
                    | ">"
                    | "<="
                    | ">="
                    | "!="
                    | "not"
                    | "and"
                    | "or"
                    | "concat"
                    | "to-string"
                    | "to-f64"
                    | "to-i64"
            ) {
                return Err(refuse(format!("non-scalar or unadmitted primitive {name}")));
            }
            for arg in args {
                validate_scalar(arg)?;
            }
            Ok(())
        }
    }
}
fn refuse(message: impl Into<String>) -> Error {
    Error::Relational(message.into())
}

#[cfg(test)]
mod negation_tests;
