//! Datalog-lite rules and predicate stratification for incremental maintenance.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

/// A ground or open atom: `predicate(arg0, arg1, ...)`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Atom {
    pub predicate: String,
    pub args: Vec<String>,
}

impl Atom {
    pub fn new(
        predicate: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            predicate: predicate.into(),
            args: args.into_iter().map(Into::into).collect(),
        }
    }

    pub fn is_ground(&self) -> bool {
        self.args.iter().all(|arg| !is_variable(arg))
    }

    pub fn fact_id(&self) -> Option<String> {
        self.is_ground()
            .then(|| format_fact(&self.predicate, &self.args))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rule {
    pub id: String,
    pub head: Atom,
    pub body: Vec<Atom>,
}

impl Rule {
    pub fn new(id: impl Into<String>, head: Atom, body: impl IntoIterator<Item = Atom>) -> Self {
        Self {
            id: id.into(),
            head,
            body: body.into_iter().collect(),
        }
    }

    pub fn body_predicates(&self) -> BTreeSet<&str> {
        self.body
            .iter()
            .map(|atom| atom.predicate.as_str())
            .collect()
    }

    pub fn head_predicate(&self) -> &str {
        &self.head.predicate
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RuleProgram {
    pub rules: Vec<Rule>,
    /// Predicates that participate in a recursive SCC. Computed at load time.
    pub recursive_predicates: BTreeSet<String>,
}

impl RuleProgram {
    pub fn from_rules(rules: Vec<Rule>) -> Self {
        let recursive_predicates = stratify_recursive_predicates(&rules);
        Self {
            rules,
            recursive_predicates,
        }
    }

    pub fn is_recursive(&self, predicate: &str) -> bool {
        self.recursive_predicates.contains(predicate)
    }

    pub fn rules_for_head(&self, predicate: &str) -> Vec<&Rule> {
        self.rules
            .iter()
            .filter(|rule| rule.head_predicate() == predicate)
            .collect()
    }
}

pub fn is_variable(arg: &str) -> bool {
    arg.chars()
        .next()
        .map(|c| c.is_uppercase() || c == '_')
        .unwrap_or(false)
}

pub fn format_fact(predicate: &str, args: &[String]) -> String {
    format!("{predicate}({})", args.join(","))
}

pub fn parse_fact(fact_id: &str) -> Option<Atom> {
    let open = fact_id.find('(')?;
    let close = fact_id.rfind(')')?;
    if close <= open {
        return None;
    }
    let predicate = fact_id[..open].to_string();
    let args_raw = &fact_id[open + 1..close];
    let args = if args_raw.is_empty() {
        Vec::new()
    } else {
        args_raw
            .split(',')
            .map(|part| part.trim().to_string())
            .collect()
    };
    Some(Atom { predicate, args })
}

pub fn fact_predicate(fact_id: &str) -> Option<String> {
    parse_fact(fact_id).map(|atom| atom.predicate)
}

/// Compute predicates that lie on a cycle in the rule dependency graph.
///
/// Edge `body_pred -> head_pred` for every body atom of every rule. A predicate
/// is recursive when its SCC has more than one node, or a self-loop.
pub fn stratify_recursive_predicates(rules: &[Rule]) -> BTreeSet<String> {
    let mut successors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut nodes: BTreeSet<String> = BTreeSet::new();
    for rule in rules {
        let head = rule.head_predicate().to_string();
        nodes.insert(head.clone());
        for body in &rule.body {
            nodes.insert(body.predicate.clone());
            successors
                .entry(body.predicate.clone())
                .or_default()
                .insert(head.clone());
        }
        successors.entry(head).or_default();
    }

    // Tarjan SCC
    struct Tarjan<'a> {
        successors: &'a BTreeMap<String, BTreeSet<String>>,
        index: usize,
        stack: Vec<String>,
        on_stack: BTreeSet<String>,
        indices: BTreeMap<String, usize>,
        lowlink: BTreeMap<String, usize>,
        sccs: Vec<Vec<String>>,
    }

    impl Tarjan<'_> {
        fn strongconnect(&mut self, v: &str) {
            self.indices.insert(v.to_string(), self.index);
            self.lowlink.insert(v.to_string(), self.index);
            self.index += 1;
            self.stack.push(v.to_string());
            self.on_stack.insert(v.to_string());

            if let Some(neighbors) = self.successors.get(v).cloned() {
                for w in neighbors {
                    if !self.indices.contains_key(&w) {
                        self.strongconnect(&w);
                        let lw = *self.lowlink.get(&w).unwrap_or(&0);
                        let lv = *self.lowlink.get(v).unwrap_or(&0);
                        self.lowlink.insert(v.to_string(), lv.min(lw));
                    } else if self.on_stack.contains(&w) {
                        let iw = *self.indices.get(&w).unwrap_or(&0);
                        let lv = *self.lowlink.get(v).unwrap_or(&0);
                        self.lowlink.insert(v.to_string(), lv.min(iw));
                    }
                }
            }

            if self.lowlink.get(v) == self.indices.get(v) {
                let mut component = Vec::new();
                loop {
                    let w = self.stack.pop().expect("scc stack");
                    self.on_stack.remove(&w);
                    component.push(w.clone());
                    if w == v {
                        break;
                    }
                }
                self.sccs.push(component);
            }
        }
    }

    let mut tarjan = Tarjan {
        successors: &successors,
        index: 0,
        stack: Vec::new(),
        on_stack: BTreeSet::new(),
        indices: BTreeMap::new(),
        lowlink: BTreeMap::new(),
        sccs: Vec::new(),
    };

    for node in &nodes {
        if !tarjan.indices.contains_key(node) {
            tarjan.strongconnect(node);
        }
    }

    let mut recursive = BTreeSet::new();
    for component in tarjan.sccs {
        let has_self_loop = component.len() == 1
            && successors
                .get(&component[0])
                .map(|set| set.contains(&component[0]))
                .unwrap_or(false);
        if component.len() > 1 || has_self_loop {
            recursive.extend(component);
        }
    }
    recursive
}

/// Match a ground fact against an open atom under a substitution.
pub fn unify_atom(pattern: &Atom, ground: &Atom, subst: &mut BTreeMap<String, String>) -> bool {
    if pattern.predicate != ground.predicate || pattern.args.len() != ground.args.len() {
        return false;
    }
    let mut local = subst.clone();
    for (pat, val) in pattern.args.iter().zip(ground.args.iter()) {
        if is_variable(pat) {
            if let Some(bound) = local.get(pat) {
                if bound != val {
                    return false;
                }
            } else {
                local.insert(pat.clone(), val.clone());
            }
        } else if pat != val {
            return false;
        }
    }
    *subst = local;
    true
}

pub fn instantiate(atom: &Atom, subst: &BTreeMap<String, String>) -> Option<Atom> {
    let mut args = Vec::with_capacity(atom.args.len());
    for arg in &atom.args {
        if is_variable(arg) {
            args.push(subst.get(arg)?.clone());
        } else {
            args.push(arg.clone());
        }
    }
    Some(Atom {
        predicate: atom.predicate.clone(),
        args,
    })
}

/// Semi-naive style: find all ground heads justified by `delta_fact` joined with
/// the current present fact set.
pub fn fire_rule_with_delta(
    rule: &Rule,
    delta_fact: &Atom,
    present: &BTreeSet<String>,
) -> Vec<(String, Vec<String>)> {
    let mut results = Vec::new();
    for body_idx in 0..rule.body.len() {
        let mut subst = BTreeMap::new();
        if !unify_atom(&rule.body[body_idx], delta_fact, &mut subst) {
            continue;
        }
        let mut join_queue: VecDeque<(usize, BTreeMap<String, String>, Vec<String>)> =
            VecDeque::new();
        join_queue.push_back((
            0,
            subst,
            vec![format_fact(&delta_fact.predicate, &delta_fact.args)],
        ));

        while let Some((next_body, subst, mut premises)) = join_queue.pop_front() {
            if next_body == rule.body.len() {
                if let Some(head) = instantiate(&rule.head, &subst) {
                    if let Some(fact_id) = head.fact_id() {
                        premises.sort();
                        premises.dedup();
                        results.push((fact_id, premises));
                    }
                }
                continue;
            }
            if next_body == body_idx {
                join_queue.push_back((next_body + 1, subst, premises));
                continue;
            }
            let pattern = &rule.body[next_body];
            for candidate in present {
                let Some(ground) = parse_fact(candidate) else {
                    continue;
                };
                let mut next_subst = subst.clone();
                if unify_atom(pattern, &ground, &mut next_subst) {
                    let mut next_premises = premises.clone();
                    next_premises.push(candidate.clone());
                    join_queue.push_back((next_body + 1, next_subst, next_premises));
                }
            }
        }
    }
    results.sort();
    results.dedup();
    results
}

/// Exhaustive join (used for DRed re-derive seeding).
pub fn fire_rule_full(rule: &Rule, present: &BTreeSet<String>) -> Vec<(String, Vec<String>)> {
    let mut results = Vec::new();
    let mut queue: VecDeque<(usize, BTreeMap<String, String>, Vec<String>)> = VecDeque::new();
    queue.push_back((0, BTreeMap::new(), Vec::new()));
    while let Some((body_idx, subst, premises)) = queue.pop_front() {
        if body_idx == rule.body.len() {
            if let Some(head) = instantiate(&rule.head, &subst) {
                if let Some(fact_id) = head.fact_id() {
                    let mut premises = premises;
                    premises.sort();
                    premises.dedup();
                    results.push((fact_id, premises));
                }
            }
            continue;
        }
        let pattern = &rule.body[body_idx];
        for candidate in present {
            let Some(ground) = parse_fact(candidate) else {
                continue;
            };
            let mut next_subst = subst.clone();
            if unify_atom(pattern, &ground, &mut next_subst) {
                let mut next_premises = premises.clone();
                next_premises.push(candidate.clone());
                queue.push_back((body_idx + 1, next_subst, next_premises));
            }
        }
    }
    results.sort();
    results.dedup();
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_rules_mark_path_recursive() {
        let program = RuleProgram::from_rules(vec![
            Rule::new(
                "path_base",
                Atom::new("path", ["X", "Y"]),
                [Atom::new("edge", ["X", "Y"])],
            ),
            Rule::new(
                "path_trans",
                Atom::new("path", ["X", "Y"]),
                [Atom::new("path", ["X", "Z"]), Atom::new("path", ["Z", "Y"])],
            ),
        ]);
        assert!(program.is_recursive("path"));
        assert!(!program.is_recursive("edge"));
    }
}
