use crate::Error;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Atom {
    String(String),
    I64(i64),
    F64(f64),
    Bool(bool),
}
impl Atom {
    pub(crate) fn literal(&self) -> Result<egglog::ast::Literal, Error> {
        Ok(match self {
            Self::String(value) => egglog::ast::Literal::String(value.clone()),
            Self::I64(value) => egglog::ast::Literal::Int(*value),
            Self::F64(value) if value.is_finite() => egglog::ast::Literal::Float((*value).into()),
            Self::F64(_) => return Err(Error::Invalid("non-finite relational float".into())),
            Self::Bool(value) => egglog::ast::Literal::Bool(*value),
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TermValue {
    Atom(Atom),
    Unit,
    Var(String),
    App(String, Vec<TermValue>),
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Term {
    /// The pinned engine's exact renderer output, retained for byte-stable goldens.
    pub expression: String,
    pub value: TermValue,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionRow {
    pub inputs: Vec<Term>,
    pub output: Term,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FunctionRows {
    pub output_sort: String,
    pub rows: Vec<FunctionRow>,
}

pub(crate) fn decode(dag: &egglog::TermDag, id: egglog::TermId) -> Term {
    Term {
        expression: dag.to_string(id),
        value: decode_value(dag, id),
    }
}
fn decode_value(dag: &egglog::TermDag, id: egglog::TermId) -> TermValue {
    use egglog::ast::Literal;
    match dag.get(id) {
        egglog::Term::Lit(value) => match value {
            Literal::String(x) => TermValue::Atom(Atom::String(x.clone())),
            Literal::Int(x) => TermValue::Atom(Atom::I64(*x)),
            Literal::Float(x) => TermValue::Atom(Atom::F64(**x)),
            Literal::Bool(x) => TermValue::Atom(Atom::Bool(*x)),
            Literal::Unit => TermValue::Unit,
        },
        egglog::Term::Var(x) => TermValue::Var(x.clone()),
        egglog::Term::App(name, children) => TermValue::App(
            name.clone(),
            children.iter().map(|id| decode_value(dag, *id)).collect(),
        ),
    }
}
