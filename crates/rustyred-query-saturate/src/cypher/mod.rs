//! Shared authoritative Cypher syntax; HTTP adapters translate the data-only error.
pub mod ast;
pub mod parse;
#[derive(Clone, Debug)]
pub struct CypherParseError {
    pub code: String,
    pub message: String,
}
impl CypherParseError {
    pub fn invalid(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
    pub fn missing_param(name: &str) -> Self {
        Self::invalid(
            "missing_cypher_param",
            format!("missing required cypher parameter ${name}"),
        )
    }
}
impl std::fmt::Display for CypherParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}
impl std::error::Error for CypherParseError {}
