use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuerySpan {
    pub start: usize,
    pub end: usize,
    pub excerpt: String,
}

#[derive(Clone, Debug)]
pub struct CypherError {
    pub code: String,
    pub message: String,
    pub span: Option<QuerySpan>,
    pub nearest: Option<String>,
    pub supported_alternative: Option<String>,
}

impl CypherError {
    pub fn invalid(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            span: None,
            nearest: None,
            supported_alternative: None,
        }
    }

    pub fn missing_param(name: &str) -> Self {
        Self::invalid(
            "missing_cypher_param",
            format!("missing required cypher parameter ${name}"),
        )
    }

    pub fn unsupported(feature: impl AsRef<str>, message: impl Into<String>) -> Self {
        Self::invalid(
            "unsupported_construct",
            format!("{}: {}", feature.as_ref(), message.into()),
        )
    }

    pub fn with_span(mut self, span: QuerySpan) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_nearest(mut self, nearest: impl Into<String>) -> Self {
        self.nearest = Some(nearest.into());
        self
    }

    pub fn with_alternative(mut self, alternative: impl Into<String>) -> Self {
        self.supported_alternative = Some(alternative.into());
        self
    }
}

impl From<rustyred_thg_core::GraphStoreError> for CypherError {
    fn from(error: rustyred_thg_core::GraphStoreError) -> Self {
        Self::invalid(error.code, error.message)
    }
}

pub fn pest_span(input: &str, err: &pest::error::Error<impl pest::RuleType>) -> QuerySpan {
    use pest::error::InputLocation;
    let (start, end) = match err.location {
        InputLocation::Pos(pos) => (pos, (pos + 1).min(input.len())),
        InputLocation::Span((a, b)) => (a, b.min(input.len())),
    };
    let excerpt = input.get(start..end).unwrap_or("").to_string();
    QuerySpan {
        start,
        end,
        excerpt,
    }
}

pub fn json_span(span: &QuerySpan) -> Value {
    serde_json::json!({
        "start": span.start,
        "end": span.end,
        "excerpt": span.excerpt,
    })
}
