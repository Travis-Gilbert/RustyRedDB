use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::errors::{ThgError, ThgResult};
use crate::state::{ThgEdge, ThgNode};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ThgCommand {
    RunBegin,
    RunStep,
    RunGet,
    ToolSelect,
    ContextPack,
    ContextGet,
    PatchPropose,
    PatchValidate,
    PatchCommit,
    StateHash,
    CypherDebug,
    GraphNodeUpsert,
    GraphEdgeUpsert,
    GraphNodesQuery,
    GraphNeighbors,
    GraphStats,
    GraphVerify,
    GraphRebuildIndexes,
    AdaptersUpsert,
    AdaptersFind,
    AdaptersGet,
    AdaptersFitnessRecord,
    AdaptersList,
    AdaptersSupersede,
    StreamPublish,
    StreamRead,
    StreamAck,
    StreamSubscribe,
    StreamUnsubscribe,
    StreamMentions,
}

pub const ALL_THG_COMMANDS: [ThgCommand; 30] = [
    ThgCommand::RunBegin,
    ThgCommand::RunStep,
    ThgCommand::RunGet,
    ThgCommand::ToolSelect,
    ThgCommand::ContextPack,
    ThgCommand::ContextGet,
    ThgCommand::PatchPropose,
    ThgCommand::PatchValidate,
    ThgCommand::PatchCommit,
    ThgCommand::StateHash,
    ThgCommand::CypherDebug,
    ThgCommand::GraphNodeUpsert,
    ThgCommand::GraphEdgeUpsert,
    ThgCommand::GraphNodesQuery,
    ThgCommand::GraphNeighbors,
    ThgCommand::GraphStats,
    ThgCommand::GraphVerify,
    ThgCommand::GraphRebuildIndexes,
    ThgCommand::AdaptersUpsert,
    ThgCommand::AdaptersFind,
    ThgCommand::AdaptersGet,
    ThgCommand::AdaptersFitnessRecord,
    ThgCommand::AdaptersList,
    ThgCommand::AdaptersSupersede,
    ThgCommand::StreamPublish,
    ThgCommand::StreamRead,
    ThgCommand::StreamAck,
    ThgCommand::StreamSubscribe,
    ThgCommand::StreamUnsubscribe,
    ThgCommand::StreamMentions,
];

impl Serialize for ThgCommand {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for ThgCommand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let name = String::deserialize(deserializer)?;
        Self::from_name(&name).map_err(|error| D::Error::custom(error.message))
    }
}

impl ThgCommand {
    pub fn from_name(name: &str) -> Result<Self, ThgError> {
        match name.trim().to_ascii_uppercase().as_str() {
            "RUSTYRED_THG.RUN.BEGIN" => Ok(Self::RunBegin),
            "RUSTYRED_THG.RUN.STEP" => Ok(Self::RunStep),
            "RUSTYRED_THG.RUN.GET" => Ok(Self::RunGet),
            "RUSTYRED_THG.TOOL.SELECT" => Ok(Self::ToolSelect),
            "RUSTYRED_THG.CONTEXT.PACK" => Ok(Self::ContextPack),
            "RUSTYRED_THG.CONTEXT.GET" => Ok(Self::ContextGet),
            "RUSTYRED_THG.PATCH.PROPOSE" => Ok(Self::PatchPropose),
            "RUSTYRED_THG.PATCH.VALIDATE" => Ok(Self::PatchValidate),
            "RUSTYRED_THG.PATCH.COMMIT" => Ok(Self::PatchCommit),
            "RUSTYRED_THG.STATE.HASH" => Ok(Self::StateHash),
            "RUSTYRED_THG.DEBUG.CYPHER" | "RUSTYRED_THG.CYPHER" => Ok(Self::CypherDebug),
            "RUSTYRED_THG.GRAPH.NODE.UPSERT" => Ok(Self::GraphNodeUpsert),
            "RUSTYRED_THG.GRAPH.EDGE.UPSERT" => Ok(Self::GraphEdgeUpsert),
            "RUSTYRED_THG.GRAPH.NODES.QUERY" => Ok(Self::GraphNodesQuery),
            "RUSTYRED_THG.GRAPH.NEIGHBORS" => Ok(Self::GraphNeighbors),
            "RUSTYRED_THG.GRAPH.STATS" => Ok(Self::GraphStats),
            "RUSTYRED_THG.GRAPH.VERIFY" => Ok(Self::GraphVerify),
            "RUSTYRED_THG.GRAPH.REBUILD_INDEXES" | "RUSTYRED_THG.GRAPH.REBUILD" => {
                Ok(Self::GraphRebuildIndexes)
            }
            "RUSTYRED_THG.ADAPTERS.UPSERT" => Ok(Self::AdaptersUpsert),
            "RUSTYRED_THG.ADAPTERS.FIND" => Ok(Self::AdaptersFind),
            "RUSTYRED_THG.ADAPTERS.GET" => Ok(Self::AdaptersGet),
            "RUSTYRED_THG.ADAPTERS.FITNESS.RECORD" => Ok(Self::AdaptersFitnessRecord),
            "RUSTYRED_THG.ADAPTERS.LIST" => Ok(Self::AdaptersList),
            "RUSTYRED_THG.ADAPTERS.SUPERSEDE" => Ok(Self::AdaptersSupersede),
            "RUSTYRED_THG.STREAM.PUBLISH" => Ok(Self::StreamPublish),
            "RUSTYRED_THG.STREAM.READ" => Ok(Self::StreamRead),
            "RUSTYRED_THG.STREAM.ACK" => Ok(Self::StreamAck),
            "RUSTYRED_THG.STREAM.SUBSCRIBE" => Ok(Self::StreamSubscribe),
            "RUSTYRED_THG.STREAM.UNSUBSCRIBE" => Ok(Self::StreamUnsubscribe),
            "RUSTYRED_THG.STREAM.MENTIONS" => Ok(Self::StreamMentions),
            _ => Err(ThgError::unsupported_command(name)),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::RunBegin => "RUSTYRED_THG.RUN.BEGIN",
            Self::RunStep => "RUSTYRED_THG.RUN.STEP",
            Self::RunGet => "RUSTYRED_THG.RUN.GET",
            Self::ToolSelect => "RUSTYRED_THG.TOOL.SELECT",
            Self::ContextPack => "RUSTYRED_THG.CONTEXT.PACK",
            Self::ContextGet => "RUSTYRED_THG.CONTEXT.GET",
            Self::PatchPropose => "RUSTYRED_THG.PATCH.PROPOSE",
            Self::PatchValidate => "RUSTYRED_THG.PATCH.VALIDATE",
            Self::PatchCommit => "RUSTYRED_THG.PATCH.COMMIT",
            Self::StateHash => "RUSTYRED_THG.STATE.HASH",
            Self::CypherDebug => "RUSTYRED_THG.DEBUG.CYPHER",
            Self::GraphNodeUpsert => "RUSTYRED_THG.GRAPH.NODE.UPSERT",
            Self::GraphEdgeUpsert => "RUSTYRED_THG.GRAPH.EDGE.UPSERT",
            Self::GraphNodesQuery => "RUSTYRED_THG.GRAPH.NODES.QUERY",
            Self::GraphNeighbors => "RUSTYRED_THG.GRAPH.NEIGHBORS",
            Self::GraphStats => "RUSTYRED_THG.GRAPH.STATS",
            Self::GraphVerify => "RUSTYRED_THG.GRAPH.VERIFY",
            Self::GraphRebuildIndexes => "RUSTYRED_THG.GRAPH.REBUILD_INDEXES",
            Self::AdaptersUpsert => "RUSTYRED_THG.ADAPTERS.UPSERT",
            Self::AdaptersFind => "RUSTYRED_THG.ADAPTERS.FIND",
            Self::AdaptersGet => "RUSTYRED_THG.ADAPTERS.GET",
            Self::AdaptersFitnessRecord => "RUSTYRED_THG.ADAPTERS.FITNESS.RECORD",
            Self::AdaptersList => "RUSTYRED_THG.ADAPTERS.LIST",
            Self::AdaptersSupersede => "RUSTYRED_THG.ADAPTERS.SUPERSEDE",
            Self::StreamPublish => "RUSTYRED_THG.STREAM.PUBLISH",
            Self::StreamRead => "RUSTYRED_THG.STREAM.READ",
            Self::StreamAck => "RUSTYRED_THG.STREAM.ACK",
            Self::StreamSubscribe => "RUSTYRED_THG.STREAM.SUBSCRIBE",
            Self::StreamUnsubscribe => "RUSTYRED_THG.STREAM.UNSUBSCRIBE",
            Self::StreamMentions => "RUSTYRED_THG.STREAM.MENTIONS",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ThgRequest {
    pub command: String,
    #[serde(default, alias = "payload")]
    pub args: Value,
}

impl ThgRequest {
    pub fn new(command: impl Into<String>, args: Value) -> Self {
        Self {
            command: command.into(),
            args,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ThgResponse {
    pub ok: bool,
    pub command: String,
    pub status: String,
    pub payload: Value,
    pub nodes: Vec<ThgNode>,
    pub edges: Vec<ThgEdge>,
    pub events: Vec<Value>,
    pub state_hash: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ThgError>,
}

impl ThgResponse {
    pub fn ok(
        command: impl Into<String>,
        status: impl Into<String>,
        payload: Value,
        state_hash: impl Into<String>,
    ) -> Self {
        Self {
            ok: true,
            command: command.into(),
            status: status.into(),
            payload,
            nodes: Vec::new(),
            edges: Vec::new(),
            events: Vec::new(),
            state_hash: state_hash.into(),
            error: None,
        }
    }

    pub fn err(command: impl Into<String>, error: ThgError, state_hash: impl Into<String>) -> Self {
        Self {
            ok: false,
            command: command.into(),
            status: error.code.clone(),
            payload: Value::Object(Default::default()),
            nodes: Vec::new(),
            edges: Vec::new(),
            events: Vec::new(),
            state_hash: state_hash.into(),
            error: Some(error),
        }
    }

    pub(crate) fn try_ok(
        command: impl Into<String>,
        status: impl Into<String>,
        payload: Value,
        state_hash: ThgResult<String>,
    ) -> ThgResult<Self> {
        Ok(Self::ok(command, status, payload, state_hash?))
    }

    pub(crate) fn try_err(
        command: impl Into<String>,
        error: ThgError,
        state_hash: ThgResult<String>,
    ) -> ThgResult<Self> {
        Ok(Self::err(command, error, state_hash?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_name_parser_and_serde_share_one_wire_encoding() {
        for command in ALL_THG_COMMANDS {
            assert_eq!(ThgCommand::from_name(command.name()).unwrap(), command);
            let encoded = serde_json::to_string(&command).unwrap();
            assert_eq!(encoded, format!("\"{}\"", command.name()));
            assert_eq!(
                serde_json::from_str::<ThgCommand>(&encoded).unwrap(),
                command
            );
        }
    }
}
