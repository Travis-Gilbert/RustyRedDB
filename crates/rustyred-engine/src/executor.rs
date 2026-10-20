use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// ponytail: avoids intermediate Vec allocation from split_whitespace().collect::<Vec<_>>().join()
fn collapse_ws(s: &str, sep: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for w in s.split_whitespace() {
        if !out.is_empty() {
            out.push_str(sep);
        }
        out.push_str(w);
    }
    out
}

use serde_json::{json, Value};

use crate::commands::{ThgCommand, ThgRequest, ThgResponse};
use crate::errors::{ThgError, ThgResult};
use crate::graph::error::GraphStoreError;
use crate::graph::in_memory::InMemoryGraphStore;
use crate::graph::query::{NeighborQuery, NodeQuery};
use crate::graph::record::{EdgeRecord, NodeRecord};
use crate::state::{
    stable_hash, ContextState, PatchState, RunState, StepState, ThgEdge, ThgNode, ThgState,
};
use crate::store::ThgStore;
use crate::stream::{
    AckPolicy, StreamStore, StreamUrgency, DEFAULT_MAX_DELIVER, DEFAULT_REDELIVER_AFTER_MS,
};

pub trait ThgExecutor {
    fn execute(&mut self, command: ThgCommand, args: Value) -> ThgResult<ThgResponse>;
    fn execute_request(&mut self, request: ThgRequest) -> ThgResult<ThgResponse>;
    fn state(&self) -> &ThgState;
}

#[derive(Clone, Debug, Default)]
pub struct InMemoryThgExecutor {
    state: Arc<ThgState>,
    graph_store: InMemoryGraphStore,
}

impl InMemoryThgExecutor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_state(state: ThgState) -> Self {
        Self::from_state_snapshot(Arc::new(state))
    }

    pub fn from_state_snapshot(state: Arc<ThgState>) -> Self {
        Self {
            state,
            graph_store: InMemoryGraphStore::new(),
        }
    }

    pub fn state_hash(&self) -> ThgResult<String> {
        self.state.hash().map_err(Into::into)
    }

    fn state_mut(&mut self) -> &mut ThgState {
        Arc::make_mut(&mut self.state)
    }

    pub fn execute_json(&mut self, request_json: &str) -> ThgResult<String> {
        match serde_json::from_str::<ThgRequest>(request_json) {
            Ok(request) => serialize_response(&self.execute_request(request)?),
            Err(exc) => {
                let response = ThgResponse::try_err(
                    "RUSTYRED_THG.UNKNOWN",
                    ThgError::invalid_json(exc.to_string()),
                    self.state_hash(),
                )?;
                serialize_response(&response)
            }
        }
    }

    fn run_begin(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let seq = self.state_mut().next_seq();
        let run_id = string_arg(&args, "run_id").unwrap_or_else(|| generated_id("run", seq));
        let run = RunState {
            run_id: run_id.clone(),
            task: string_arg(&args, "task").unwrap_or_default(),
            actor: string_arg(&args, "actor").unwrap_or_else(|| "agent".to_string()),
            scope: args.get("scope").cloned().unwrap_or_else(|| json!({})),
            status: "running".to_string(),
            steps: Vec::new(),
        };
        let node = run.node();
        self.state_mut().runs.insert(run_id.clone(), run);
        let mut response = ThgResponse::try_ok(
            ThgCommand::RunBegin.name(),
            "ok",
            json!({ "run_id": run_id, "status": "running" }),
            self.state_hash(),
        )?;
        response.nodes.push(node);
        response
            .events
            .push(json!({ "event": "run_begin", "run_id": run_id }));
        Ok(response)
    }

    fn run_step(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let run_id = string_arg(&args, "run_id").unwrap_or_default();
        if !self.state.runs.contains_key(&run_id) {
            return ThgResponse::try_err(
                ThgCommand::RunStep.name(),
                ThgError::new("unknown_run_id", format!("run {run_id} does not exist")),
                self.state_hash(),
            );
        }
        let seq = self.state_mut().next_seq();
        let step_id = string_arg(&args, "step_id").unwrap_or_else(|| generated_id("step", seq));
        let index = int_arg(&args, "index").unwrap_or_else(|| {
            self.state
                .runs
                .get(&run_id)
                .map(|run| run.steps.len() as i64 + 1)
                .unwrap_or(1)
        });
        let step = StepState {
            step_id: step_id.clone(),
            kind: string_arg(&args, "kind").unwrap_or_else(|| "observation".to_string()),
            index,
            payload: args.get("payload").cloned().unwrap_or_else(|| json!({})),
        };
        let node = step.node(&run_id);
        self.state_mut()
            .runs
            .get_mut(&run_id)
            .expect("run existence checked above")
            .steps
            .push(step);
        let edge = ThgEdge {
            from_id: run_id.clone(),
            edge_type: "HAS_STEP".to_string(),
            to_id: step_id.clone(),
            properties: json!({ "index": index }),
        };
        let mut response = ThgResponse::try_ok(
            ThgCommand::RunStep.name(),
            "ok",
            json!({ "run_id": run_id, "step_id": step_id }),
            self.state_hash(),
        )?;
        response.nodes.push(node);
        response.edges.push(edge);
        response
            .events
            .push(json!({ "event": "run_step", "run_id": run_id, "step_id": step_id }));
        Ok(response)
    }

    fn run_get(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let run_id = string_arg(&args, "run_id").unwrap_or_default();
        let run = self.state.runs.get(&run_id).cloned();
        ThgResponse::try_ok(
            ThgCommand::RunGet.name(),
            if run.is_some() { "ok" } else { "not_found" },
            json!({ "run": run }),
            self.state_hash(),
        )
    }

    fn tool_select(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let task_type = string_arg(&args, "task_type").unwrap_or_else(|| "other".to_string());
        let required = string_vec_arg(&args, "required_skills");
        let toolkit = compile_toolkit(&task_type, &required);
        let mut response = ThgResponse::try_ok(
            ThgCommand::ToolSelect.name(),
            "ok",
            json!({ "task_type": task_type, "toolkit": toolkit }),
            self.state_hash(),
        )?;
        response.nodes.push(ThgNode {
            id: format!("tasktype:{task_type}"),
            labels: vec!["TaskType".to_string()],
            properties: json!({ "task_type": task_type }),
        });
        for tool in toolkit {
            let tool_id = tool
                .get("tool_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            response.nodes.push(ThgNode {
                id: tool_id.clone(),
                labels: vec!["Tool".to_string()],
                properties: tool,
            });
            response.edges.push(ThgEdge {
                from_id: format!("tasktype:{task_type}"),
                edge_type: "COMPILES_TOOL".to_string(),
                to_id: tool_id,
                properties: json!({}),
            });
        }
        Ok(response)
    }

    fn context_pack(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let seq = self.state_mut().next_seq();
        let artifact_id =
            string_arg(&args, "artifact_id").unwrap_or_else(|| generated_id("artifact", seq));
        let context = ContextState {
            artifact_id: artifact_id.clone(),
            status: "packed".to_string(),
            sections: args.get("sections").cloned().unwrap_or_else(|| json!([])),
            token_ledger: args
                .get("token_ledger")
                .cloned()
                .unwrap_or_else(|| json!({})),
        };
        let node = context.node();
        self.state_mut()
            .contexts
            .insert(artifact_id.clone(), context);
        let mut response = ThgResponse::try_ok(
            ThgCommand::ContextPack.name(),
            "ok",
            json!({
                "artifact_id": artifact_id,
                "sections": args.get("sections").cloned().unwrap_or_else(|| json!([])),
                "token_ledger": args.get("token_ledger").cloned().unwrap_or_else(|| json!({})),
            }),
            self.state_hash(),
        )?;
        response.nodes.push(node);
        Ok(response)
    }

    fn context_get(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let artifact_id = string_arg(&args, "artifact_id").unwrap_or_default();
        let context = self.state.contexts.get(&artifact_id).cloned();
        ThgResponse::try_ok(
            ThgCommand::ContextGet.name(),
            if context.is_some() { "ok" } else { "not_found" },
            json!({ "context": context }),
            self.state_hash(),
        )
    }

    fn patch_propose(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let seq = self.state_mut().next_seq();
        let patch_id = string_arg(&args, "patch_id").unwrap_or_else(|| generated_id("patch", seq));
        let run_id = string_arg(&args, "run_id").unwrap_or_default();
        let patch = PatchState {
            patch_id: patch_id.clone(),
            run_id: run_id.clone(),
            status: "proposed".to_string(),
            patch: args.get("patch").cloned().unwrap_or_else(|| json!({})),
            findings: json!([]),
        };
        let node = patch.node();
        self.state_mut().patches.insert(patch_id.clone(), patch);
        let mut response = ThgResponse::try_ok(
            ThgCommand::PatchPropose.name(),
            "ok",
            json!({ "patch_id": patch_id, "status": "proposed" }),
            self.state_hash(),
        )?;
        response.nodes.push(node);
        if !run_id.is_empty() {
            response.edges.push(ThgEdge {
                from_id: run_id,
                edge_type: "PROPOSED_PATCH".to_string(),
                to_id: patch_id,
                properties: json!({}),
            });
        }
        Ok(response)
    }

    fn patch_validate(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let patch_id = string_arg(&args, "patch_id").unwrap_or_default();
        let findings = args
            .get("findings")
            .cloned()
            .unwrap_or_else(|| json!([{ "code": "human_review_required" }]));
        let Some(patch) = self.state_mut().patches.get_mut(&patch_id) else {
            return ThgResponse::try_err(
                ThgCommand::PatchValidate.name(),
                ThgError::new(
                    "unknown_patch_id",
                    format!("patch {patch_id} does not exist"),
                ),
                self.state_hash(),
            );
        };
        patch.status = "needs_review".to_string();
        patch.findings = findings.clone();
        ThgResponse::try_ok(
            ThgCommand::PatchValidate.name(),
            "ok",
            json!({ "patch_id": patch_id, "status": "needs_review", "findings": findings }),
            self.state_hash(),
        )
    }

    fn patch_commit(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let patch_id = string_arg(&args, "patch_id").unwrap_or_default();
        let Some(patch) = self.state_mut().patches.get_mut(&patch_id) else {
            return ThgResponse::try_err(
                ThgCommand::PatchCommit.name(),
                ThgError::new(
                    "unknown_patch_id",
                    format!("patch {patch_id} does not exist"),
                ),
                self.state_hash(),
            );
        };
        patch.status = "committed".to_string();
        ThgResponse::try_ok(
            ThgCommand::PatchCommit.name(),
            "ok",
            json!({ "patch_id": patch_id, "status": "committed" }),
            self.state_hash(),
        )
    }

    fn state_hash_command(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let hash = match args.get("state") {
            Some(value) => stable_hash(value).map_err(ThgError::from),
            None => self.state_hash(),
        }?;
        ThgResponse::try_ok(
            ThgCommand::StateHash.name(),
            "ok",
            json!({ "hash": hash }),
            self.state_hash(),
        )
    }

    fn cypher_debug(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let graph = args.get("graph").cloned().unwrap_or_else(|| {
            let (nodes, edges) = self.state.graph();
            json!({ "nodes": nodes, "edges": edges })
        });
        let query = string_arg(&args, "query").unwrap_or_default();
        let rows = debug_cypher_rows(&query, &graph);
        ThgResponse::try_ok(
            ThgCommand::CypherDebug.name(),
            "ok",
            json!({ "rows": rows, "row_count": rows.as_array().map(Vec::len).unwrap_or(0) }),
            self.state_hash(),
        )
    }

    fn graph_node_upsert(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let command = ThgCommand::GraphNodeUpsert.name();
        self.state_mut().next_seq();
        let node = match node_record_from_args(args) {
            Ok(node) => node,
            Err(error) => return ThgResponse::try_err(command, error, self.state_hash()),
        };
        let response_node = thg_node_from_record(&node);
        match self.graph_store.upsert_node(node) {
            Ok(write) => {
                let mut response = ThgResponse::try_ok(
                    command,
                    "ok",
                    json!({ "write": write, "node": response_node }),
                    self.state_hash(),
                )?;
                response.nodes.push(response_node);
                Ok(response)
            }
            Err(error) => graph_store_response_error(command, error, self.state_hash()),
        }
    }

    fn graph_edge_upsert(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let command = ThgCommand::GraphEdgeUpsert.name();
        self.state_mut().next_seq();
        let edge = match edge_record_from_args(args) {
            Ok(edge) => edge,
            Err(error) => return ThgResponse::try_err(command, error, self.state_hash()),
        };
        let response_edge = thg_edge_from_record(&edge);
        match self.graph_store.upsert_edge(edge) {
            Ok(write) => {
                let mut response = ThgResponse::try_ok(
                    command,
                    "ok",
                    json!({ "write": write, "edge": response_edge }),
                    self.state_hash(),
                )?;
                response.edges.push(response_edge);
                Ok(response)
            }
            Err(error) => graph_store_response_error(command, error, self.state_hash()),
        }
    }

    fn graph_nodes_query(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let command = ThgCommand::GraphNodesQuery.name();
        let query = match serde_json::from_value::<NodeQuery>(args) {
            Ok(query) => query,
            Err(error) => {
                return ThgResponse::try_err(
                    command,
                    ThgError::new("invalid_graph_query", error.to_string()),
                    self.state_hash(),
                );
            }
        };
        let operation = if query.label.is_some() || !query.properties.is_empty() {
            "node_index_seek"
        } else {
            "node_scan"
        };
        let hits = self.graph_store.query_nodes(query);
        let nodes = hits
            .iter()
            .map(thg_node_from_record)
            .collect::<Vec<ThgNode>>();
        let mut response = ThgResponse::try_ok(
            command,
            "ok",
            json!({
                "nodes": hits,
                "plan": { "operation": operation },
                "stats": { "returned": nodes.len() },
            }),
            self.state_hash(),
        )?;
        response.nodes = nodes;
        Ok(response)
    }

    fn graph_neighbors(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let command = ThgCommand::GraphNeighbors.name();
        let query = match serde_json::from_value::<NeighborQuery>(args) {
            Ok(query) => query,
            Err(error) => {
                return ThgResponse::try_err(
                    command,
                    ThgError::new("invalid_graph_query", error.to_string()),
                    self.state_hash(),
                );
            }
        };
        let hits = self.graph_store.neighbors(query);
        ThgResponse::try_ok(
            command,
            "ok",
            json!({
                "neighbors": hits,
                "plan": { "operation": "adjacency_seek" },
                "stats": { "returned": hits.len() },
            }),
            self.state_hash(),
        )
    }

    fn graph_stats(&mut self) -> ThgResult<ThgResponse> {
        ThgResponse::try_ok(
            ThgCommand::GraphStats.name(),
            "ok",
            json!({ "stats": self.graph_store.stats() }),
            self.state_hash(),
        )
    }

    fn graph_verify(&mut self) -> ThgResult<ThgResponse> {
        let report = self.graph_store.verify();
        ThgResponse::try_ok(
            ThgCommand::GraphVerify.name(),
            if report.ok { "ok" } else { "drift_detected" },
            json!({ "report": report }),
            self.state_hash(),
        )
    }

    fn graph_rebuild_indexes(&mut self) -> ThgResult<ThgResponse> {
        match self.graph_store.rebuild_indexes() {
            Ok(report) => ThgResponse::try_ok(
                ThgCommand::GraphRebuildIndexes.name(),
                if report.after.ok {
                    "ok"
                } else {
                    "canonical_graph_problem"
                },
                json!({ "report": report }),
                self.state_hash(),
            ),
            Err(error) => ThgResponse::try_err(
                ThgCommand::GraphRebuildIndexes.name(),
                ThgError::new(error.code, error.message),
                self.state_hash(),
            ),
        }
    }

    fn stream_publish(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let command = ThgCommand::StreamPublish.name();
        let tenant = string_arg(&args, "tenant").unwrap_or_default();
        let topic = string_arg(&args, "stream")
            .or_else(|| string_arg(&args, "topic"))
            .unwrap_or_default();
        let actor = string_arg(&args, "actor").unwrap_or_default();
        if actor.trim().is_empty() {
            return ThgResponse::try_err(
                command,
                ThgError::new("missing_actor", "stream publish requires an actor"),
                self.state_hash(),
            );
        }
        let kind = string_arg(&args, "kind").unwrap_or_else(|| "message".to_string());
        let payload = args.get("payload").cloned().unwrap_or_else(|| json!({}));
        let urgency_raw = string_arg(&args, "urgency").unwrap_or_default();
        let urgency = match StreamUrgency::parse(&urgency_raw) {
            Some(urgency) => urgency,
            None => {
                return ThgResponse::try_err(
                    command,
                    ThgError::new(
                        "invalid_urgency",
                        format!("unknown urgency: {urgency_raw}; expected info|ask|block"),
                    ),
                    self.state_hash(),
                );
            }
        };
        let target_actor = string_arg(&args, "target_actor");
        // A ping (ask|block) must name a target to bridge to its wake path.
        if urgency.is_ping()
            && target_actor
                .as_deref()
                .map(str::trim)
                .unwrap_or("")
                .is_empty()
        {
            return ThgResponse::try_err(
                command,
                ThgError::new(
                    "missing_target_actor",
                    "urgency ask|block requires a target_actor",
                ),
                self.state_hash(),
            );
        }
        match self.state_mut().streams.publish(
            &tenant,
            &topic,
            &actor,
            &kind,
            payload,
            urgency,
            target_actor,
        ) {
            Ok(event) => {
                let event_value = serde_json::to_value(&event).unwrap_or_else(|_| json!({}));
                let mut response = ThgResponse::try_ok(
                    command,
                    "ok",
                    json!({
                        "event_id": event.id,
                        "ordering_token": event.ordering_token,
                        "stream_key": event.stream_key,
                        "urgency": event.urgency.as_str(),
                        "pinged": event.urgency.is_ping() && event.target_actor.is_some(),
                    }),
                    self.state_hash(),
                )?;
                response.events.push(event_value);
                Ok(response)
            }
            Err(error) => ThgResponse::try_err(
                command,
                ThgError::new(error.code, error.message),
                self.state_hash(),
            ),
        }
    }

    fn stream_read(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let command = ThgCommand::StreamRead.name();
        let actor = string_arg(&args, "actor").unwrap_or_default();
        if actor.trim().is_empty() {
            return ThgResponse::try_err(
                command,
                ThgError::new("missing_actor", "stream read requires an actor"),
                self.state_hash(),
            );
        }
        let tenant = string_arg(&args, "tenant").unwrap_or_default();
        let advance = bool_arg(&args, "advance").unwrap_or(true);
        let limit = int_arg(&args, "limit").unwrap_or(0).max(0) as usize;
        let ack_policy =
            match parse_ack_policy(&string_arg(&args, "ack_policy").unwrap_or_default()) {
                Ok(policy) => policy,
                Err(error) => {
                    return ThgResponse::try_err(command, error, self.state_hash());
                }
            };
        // `streams[]` are topics resolved under `tenant`; omitting them reads the
        // actor's subscription set (selective attention).
        let topics = string_vec_arg(&args, "streams");
        let mut resolved = Vec::with_capacity(topics.len());
        for topic in &topics {
            match StreamStore::resolve_stream_key(&tenant, topic) {
                Ok(key) => resolved.push(key),
                Err(error) => {
                    return ThgResponse::try_err(
                        command,
                        ThgError::new(error.code, error.message),
                        self.state_hash(),
                    );
                }
            }
        }
        if ack_policy == AckPolicy::Explicit {
            if !advance {
                let (events, new_cursors) = match self
                    .state_mut()
                    .streams
                    .read(&tenant, &actor, &resolved, false, limit)
                {
                    Ok(result) => result,
                    Err(error) => {
                        return ThgResponse::try_err(
                            command,
                            ThgError::new(error.code, error.message),
                            self.state_hash(),
                        );
                    }
                };
                let events_json: Vec<Value> = events
                    .iter()
                    .map(|event| serde_json::to_value(event).unwrap_or_else(|_| json!({})))
                    .collect();
                let mut response = ThgResponse::try_ok(
                    command,
                    "ok",
                    json!({
                        "events": events_json,
                        "new_cursors": new_cursors,
                        "count": events_json.len(),
                        "advanced": false,
                        "ack_policy": "explicit",
                        "pending": [],
                        "dead_lettered": [],
                    }),
                    self.state_hash(),
                )?;
                response.events = events_json;
                return Ok(response);
            }
            let now_ms = int_arg(&args, "now_ms")
                .and_then(|value| u64::try_from(value).ok())
                .unwrap_or_else(unix_millis);
            let redeliver_after_ms = int_arg(&args, "redeliver_after_ms")
                .and_then(|value| u64::try_from(value).ok())
                .unwrap_or(DEFAULT_REDELIVER_AFTER_MS);
            let max_deliver = int_arg(&args, "max_deliver")
                .and_then(|value| u32::try_from(value).ok())
                .unwrap_or(DEFAULT_MAX_DELIVER);
            let delivery = match self.state_mut().streams.read_delivery(
                &tenant,
                &actor,
                &resolved,
                ack_policy,
                now_ms,
                redeliver_after_ms,
                max_deliver,
                limit,
            ) {
                Ok(result) => result,
                Err(error) => {
                    return ThgResponse::try_err(
                        command,
                        ThgError::new(error.code, error.message),
                        self.state_hash(),
                    );
                }
            };
            let events_json: Vec<Value> = delivery
                .events
                .iter()
                .map(|event| serde_json::to_value(event).unwrap_or_else(|_| json!({})))
                .collect();
            let pending_json =
                serde_json::to_value(&delivery.pending).unwrap_or_else(|_| json!([]));
            let dead_lettered_json =
                serde_json::to_value(&delivery.dead_lettered).unwrap_or_else(|_| json!([]));
            let mut response = ThgResponse::try_ok(
                command,
                "ok",
                json!({
                    "events": events_json,
                    "new_cursors": delivery.cursors,
                    "count": events_json.len(),
                    "advanced": true,
                    "ack_policy": "explicit",
                    "pending": pending_json,
                    "dead_lettered": dead_lettered_json,
                }),
                self.state_hash(),
            )?;
            response.events = events_json;
            return Ok(response);
        }
        let (events, new_cursors) = match self
            .state_mut()
            .streams
            .read(&tenant, &actor, &resolved, advance, limit)
        {
            Ok(result) => result,
            Err(error) => {
                return ThgResponse::try_err(
                    command,
                    ThgError::new(error.code, error.message),
                    self.state_hash(),
                );
            }
        };
        let events_json: Vec<Value> = events
            .iter()
            .map(|event| serde_json::to_value(event).unwrap_or_else(|_| json!({})))
            .collect();
        let mut response = ThgResponse::try_ok(
            command,
            "ok",
            json!({
                "events": events_json,
                "new_cursors": new_cursors,
                "count": events_json.len(),
                "advanced": advance,
                "ack_policy": "on_read",
            }),
            self.state_hash(),
        )?;
        response.events = events_json;
        Ok(response)
    }

    fn stream_ack(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let command = ThgCommand::StreamAck.name();
        let actor = string_arg(&args, "actor").unwrap_or_default();
        if actor.trim().is_empty() {
            return ThgResponse::try_err(
                command,
                ThgError::new("missing_actor", "stream ack requires an actor"),
                self.state_hash(),
            );
        }
        let tenant = string_arg(&args, "tenant").unwrap_or_default();
        let topic = string_arg(&args, "stream")
            .or_else(|| string_arg(&args, "topic"))
            .unwrap_or_default();
        let stream_key = match StreamStore::resolve_stream_key(&tenant, &topic) {
            Ok(key) => key,
            Err(error) => {
                return ThgResponse::try_err(
                    command,
                    ThgError::new(error.code, error.message),
                    self.state_hash(),
                );
            }
        };
        let tokens = u64_vec_arg(&args, "ordering_tokens");
        if tokens.is_empty() {
            return ThgResponse::try_err(
                command,
                ThgError::new(
                    "missing_ordering_tokens",
                    "stream ack requires ordering_tokens",
                ),
                self.state_hash(),
            );
        }
        let acked = match self
            .state_mut()
            .streams
            .ack(&tenant, &actor, &stream_key, &tokens)
        {
            Ok(result) => result,
            Err(error) => {
                return ThgResponse::try_err(
                    command,
                    ThgError::new(error.code, error.message),
                    self.state_hash(),
                );
            }
        };
        ThgResponse::try_ok(
            command,
            "ok",
            json!({
                "tenant": tenant,
                "actor": actor,
                "stream": topic,
                "stream_key": stream_key,
                "ordering_tokens": tokens,
                "acked": acked,
            }),
            self.state_hash(),
        )
    }

    fn stream_set_subscription(&mut self, args: Value, subscribe: bool) -> ThgResult<ThgResponse> {
        let command = if subscribe {
            ThgCommand::StreamSubscribe.name()
        } else {
            ThgCommand::StreamUnsubscribe.name()
        };
        let actor = string_arg(&args, "actor").unwrap_or_default();
        if actor.trim().is_empty() {
            return ThgResponse::try_err(
                command,
                ThgError::new("missing_actor", "stream subscription requires an actor"),
                self.state_hash(),
            );
        }
        let tenant = string_arg(&args, "tenant").unwrap_or_default();
        let topic = string_arg(&args, "stream")
            .or_else(|| string_arg(&args, "topic"))
            .unwrap_or_default();
        let stream_key = match StreamStore::resolve_stream_key(&tenant, &topic) {
            Ok(key) => key,
            Err(error) => {
                return ThgResponse::try_err(
                    command,
                    ThgError::new(error.code, error.message),
                    self.state_hash(),
                );
            }
        };
        let subscriptions = match if subscribe {
            self.state_mut()
                .streams
                .subscribe(&tenant, &actor, &stream_key)
        } else {
            self.state_mut()
                .streams
                .unsubscribe(&tenant, &actor, &stream_key)
        } {
            Ok(subscriptions) => subscriptions,
            Err(error) => {
                return ThgResponse::try_err(
                    command,
                    ThgError::new(error.code, error.message),
                    self.state_hash(),
                );
            }
        };
        ThgResponse::try_ok(
            command,
            "ok",
            json!({
                "actor": actor,
                "stream_key": stream_key,
                "subscribed": subscribe,
                "subscriptions": subscriptions,
            }),
            self.state_hash(),
        )
    }

    fn stream_mentions(&mut self, args: Value) -> ThgResult<ThgResponse> {
        let command = ThgCommand::StreamMentions.name();
        let actor = string_arg(&args, "actor").unwrap_or_default();
        if actor.trim().is_empty() {
            return ThgResponse::try_err(
                command,
                ThgError::new("missing_actor", "stream mentions requires an actor"),
                self.state_hash(),
            );
        }
        let tenant = string_arg(&args, "tenant").unwrap_or_default();
        let advance = bool_arg(&args, "advance").unwrap_or(true);
        let ack_tokens = u64_vec_arg(&args, "ack_ordering_tokens");
        let acknowledged = match self
            .state_mut()
            .streams
            .ack_mentions(&tenant, &actor, &ack_tokens)
        {
            Ok(acknowledged) => acknowledged,
            Err(error) => {
                return ThgResponse::try_err(
                    command,
                    ThgError::new(error.code, error.message),
                    self.state_hash(),
                );
            }
        };
        let mentions = match self
            .state_mut()
            .streams
            .drain_mentions(&tenant, &actor, advance)
        {
            Ok(mentions) => mentions,
            Err(error) => {
                return ThgResponse::try_err(
                    command,
                    ThgError::new(error.code, error.message),
                    self.state_hash(),
                );
            }
        };
        let mentions_json: Vec<Value> = mentions
            .iter()
            .map(|event| serde_json::to_value(event).unwrap_or_else(|_| json!({})))
            .collect();
        let mut response = ThgResponse::try_ok(
            command,
            "ok",
            json!({
                "mentions": mentions_json,
                "count": mentions_json.len(),
                "drained": false,
                "ack_required": advance,
                "acknowledged": acknowledged,
            }),
            self.state_hash(),
        )?;
        response.events = mentions_json;
        Ok(response)
    }
}

#[derive(Clone, Debug)]
pub struct StoreBackedThgExecutor<S: ThgStore> {
    store: S,
    inner: InMemoryThgExecutor,
}

impl<S: ThgStore> StoreBackedThgExecutor<S> {
    pub fn new(store: S) -> ThgResult<Self> {
        let state = store.load_snapshot()?;
        Ok(Self {
            store,
            inner: InMemoryThgExecutor::from_state_snapshot(state),
        })
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn state_hash(&self) -> ThgResult<String> {
        self.inner.state_hash()
    }

    pub fn execute_json(&mut self, request_json: &str) -> ThgResult<String> {
        match serde_json::from_str::<ThgRequest>(request_json) {
            Ok(request) => serialize_response(&self.execute_request(request)?),
            Err(exc) => {
                let response = ThgResponse::try_err(
                    "RUSTYRED_THG.UNKNOWN",
                    ThgError::invalid_json(exc.to_string()),
                    self.state_hash(),
                )?;
                serialize_response(&response)
            }
        }
    }

    fn persist(&mut self) -> ThgResult<()> {
        self.store.save(self.inner.state())
    }
}

impl ThgExecutor for InMemoryThgExecutor {
    fn execute(&mut self, command: ThgCommand, args: Value) -> ThgResult<ThgResponse> {
        match command {
            ThgCommand::RunBegin => self.run_begin(args),
            ThgCommand::RunStep => self.run_step(args),
            ThgCommand::RunGet => self.run_get(args),
            ThgCommand::ToolSelect => self.tool_select(args),
            ThgCommand::ContextPack => self.context_pack(args),
            ThgCommand::ContextGet => self.context_get(args),
            ThgCommand::PatchPropose => self.patch_propose(args),
            ThgCommand::PatchValidate => self.patch_validate(args),
            ThgCommand::PatchCommit => self.patch_commit(args),
            ThgCommand::StateHash => self.state_hash_command(args),
            ThgCommand::CypherDebug => self.cypher_debug(args),
            ThgCommand::GraphNodeUpsert => self.graph_node_upsert(args),
            ThgCommand::GraphEdgeUpsert => self.graph_edge_upsert(args),
            ThgCommand::GraphNodesQuery => self.graph_nodes_query(args),
            ThgCommand::GraphNeighbors => self.graph_neighbors(args),
            ThgCommand::GraphStats => self.graph_stats(),
            ThgCommand::GraphVerify => self.graph_verify(),
            ThgCommand::GraphRebuildIndexes => self.graph_rebuild_indexes(),
            ThgCommand::StreamPublish => self.stream_publish(args),
            ThgCommand::StreamRead => self.stream_read(args),
            ThgCommand::StreamAck => self.stream_ack(args),
            ThgCommand::StreamSubscribe => self.stream_set_subscription(args, true),
            ThgCommand::StreamUnsubscribe => self.stream_set_subscription(args, false),
            ThgCommand::StreamMentions => self.stream_mentions(args),
            ThgCommand::AdaptersUpsert
            | ThgCommand::AdaptersFind
            | ThgCommand::AdaptersGet
            | ThgCommand::AdaptersFitnessRecord
            | ThgCommand::AdaptersList
            | ThgCommand::AdaptersSupersede => ThgResponse::try_err(
                command.name(),
                ThgError::unsupported_command(command.name()),
                self.state_hash(),
            ),
        }
    }

    fn execute_request(&mut self, request: ThgRequest) -> ThgResult<ThgResponse> {
        let command_name = request.command.clone();
        match ThgCommand::from_name(&request.command) {
            Ok(command) => self.execute(command, request.args),
            Err(error) => ThgResponse::try_err(command_name, error, self.state_hash()),
        }
    }

    fn state(&self) -> &ThgState {
        self.state.as_ref()
    }
}

impl<S: ThgStore> ThgExecutor for StoreBackedThgExecutor<S> {
    fn execute(&mut self, command: ThgCommand, args: Value) -> ThgResult<ThgResponse> {
        let prior_state = Arc::clone(&self.inner.state);
        let response = self.inner.execute(command, args)?;
        if response.ok {
            if let Err(error) = self.persist() {
                self.inner.state = prior_state;
                return Err(error);
            }
        }
        Ok(response)
    }

    fn execute_request(&mut self, request: ThgRequest) -> ThgResult<ThgResponse> {
        let command_name = request.command.clone();
        match ThgCommand::from_name(&request.command) {
            Ok(command) => self.execute(command, request.args),
            Err(error) => ThgResponse::try_err(command_name, error, self.state_hash()),
        }
    }

    fn state(&self) -> &ThgState {
        self.inner.state()
    }
}

pub fn execute_request_json(
    executor: &mut InMemoryThgExecutor,
    request_json: &str,
) -> ThgResult<String> {
    executor.execute_json(request_json)
}

fn serialize_response(response: &ThgResponse) -> ThgResult<String> {
    serde_json::to_string(response).map_err(|error| {
        ThgError::new(
            "response_serialization_failed",
            format!("failed to serialize THG response: {error}"),
        )
    })
}

fn generated_id(prefix: &str, seq: u64) -> String {
    format!("{prefix}:{seq:016x}")
}

fn string_arg(args: &Value, key: &str) -> Option<String> {
    args.get(key).and_then(Value::as_str).map(str::to_string)
}

fn bool_arg(args: &Value, key: &str) -> Option<bool> {
    args.get(key).and_then(Value::as_bool)
}

fn int_arg(args: &Value, key: &str) -> Option<i64> {
    args.get(key).and_then(Value::as_i64)
}

fn u64_vec_arg(args: &Value, key: &str) -> Vec<u64> {
    args.get(key)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default()
}

fn string_vec_arg(args: &Value, key: &str) -> Vec<String> {
    args.get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_ack_policy(value: &str) -> Result<AckPolicy, ThgError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "on_read" | "on-read" | "read" => Ok(AckPolicy::OnRead),
        "explicit" => Ok(AckPolicy::Explicit),
        _ => Err(ThgError::new(
            "invalid_ack_policy",
            "ack_policy must be on_read or explicit",
        )),
    }
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

fn compile_toolkit(task_type: &str, required: &[String]) -> Vec<Value> {
    let catalog = vec![
        json!({
            "tool_id": "native_search",
            "name": "Native Search",
            "skills": ["local_webdoc_search", "redis_priors", "graph_candidates"],
            "inputs": ["query", "scope", "budget"],
            "outputs": ["ranked_results", "search_trace_id", "graph_candidates"],
            "cost": "low",
            "permissions": ["web_browse"],
        }),
        json!({
            "tool_id": "context_artifact_compile",
            "name": "Context Artifact Compile",
            "skills": ["capsule_packing", "token_ledger", "artifact_export"],
            "inputs": ["run_id", "task", "budget_tokens"],
            "outputs": ["context_artifact", "token_ledger", "provenance"],
            "cost": "medium",
            "permissions": ["write_context_artifact"],
        }),
        json!({
            "tool_id": "memory_patch_validation",
            "name": "Memory Patch Validation",
            "skills": ["proposal_review", "provenance_check"],
            "inputs": ["run_id", "patch"],
            "outputs": ["validation_result"],
            "cost": "low",
            "permissions": ["propose_memory_patch"],
        }),
    ];
    if required.is_empty() {
        if matches!(task_type, "search" | "research" | "plan" | "fix" | "review") {
            return catalog.into_iter().take(2).collect();
        }
        return catalog.into_iter().skip(1).take(1).collect();
    }
    let required_set: std::collections::BTreeSet<String> = required.iter().cloned().collect();
    let mut selected = Vec::new();
    for mut tool in catalog {
        let matched: Vec<String> = tool
            .get("skills")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|skill| required_set.contains(*skill))
            .map(str::to_string)
            .collect();
        if !matched.is_empty() {
            tool["matched_skills"] = json!(matched);
            selected.push(tool);
        }
    }
    selected
}

fn debug_cypher_rows(query: &str, graph: &Value) -> Value {
    let normalized = collapse_ws(&query, " ");
    let nodes = graph
        .get("nodes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let edges = graph
        .get("edges")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    if normalized.starts_with("MATCH (n:") && normalized.ends_with("RETURN n") {
        let label = normalized
            .trim_start_matches("MATCH (n:")
            .split_whitespace()
            .next()
            .unwrap_or("")
            .trim_end_matches(')')
            .to_string();
        let rows: Vec<Value> = nodes
            .into_iter()
            .filter(|node| {
                node.get("labels")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .any(|item| item.as_str() == Some(label.as_str()))
            })
            .map(|node| json!({ "n": node }))
            .collect();
        return json!(rows);
    }

    if normalized.starts_with("MATCH (a)-[e:") && normalized.contains("]->(b) RETURN a, e, b") {
        let edge_type = normalized
            .trim_start_matches("MATCH (a)-[e:")
            .split(']')
            .next()
            .unwrap_or("");
        let rows: Vec<Value> = edges
            .into_iter()
            .filter(|edge| edge.get("type").and_then(Value::as_str) == Some(edge_type))
            .map(|edge| json!({ "e": edge }))
            .collect();
        return json!(rows);
    }

    json!([])
}

fn node_record_from_args(args: Value) -> Result<NodeRecord, ThgError> {
    let id = string_arg(&args, "id")
        .or_else(|| string_arg(&args, "node_id"))
        .ok_or_else(|| ThgError::new("empty_graph_field", "node.id is required"))?;
    let labels = string_vec_arg(&args, "labels");
    let properties = args.get("properties").cloned().unwrap_or_else(|| json!({}));
    let mut node = NodeRecord::new(id, labels, properties);
    node.tombstone = bool_arg(&args, "tombstone").unwrap_or(false);
    Ok(node)
}

fn edge_record_from_args(args: Value) -> Result<EdgeRecord, ThgError> {
    let id = string_arg(&args, "id")
        .or_else(|| string_arg(&args, "edge_id"))
        .ok_or_else(|| ThgError::new("empty_graph_field", "edge.id is required"))?;
    let from_id = string_arg(&args, "from_id")
        .ok_or_else(|| ThgError::new("empty_graph_field", "edge.from_id is required"))?;
    let to_id = string_arg(&args, "to_id")
        .ok_or_else(|| ThgError::new("empty_graph_field", "edge.to_id is required"))?;
    let edge_type = string_arg(&args, "type")
        .or_else(|| string_arg(&args, "edge_type"))
        .ok_or_else(|| ThgError::new("empty_graph_field", "edge.type is required"))?;
    let properties = args.get("properties").cloned().unwrap_or_else(|| json!({}));
    let mut edge = EdgeRecord::new(id, from_id, edge_type, to_id, properties);
    edge.tombstone = bool_arg(&args, "tombstone").unwrap_or(false);
    Ok(edge)
}

fn thg_node_from_record(node: &NodeRecord) -> ThgNode {
    ThgNode {
        id: node.id.clone(),
        labels: node.labels.clone(),
        properties: node.properties.clone(),
    }
}

fn thg_edge_from_record(edge: &EdgeRecord) -> ThgEdge {
    ThgEdge {
        from_id: edge.from_id.clone(),
        edge_type: edge.edge_type.clone(),
        to_id: edge.to_id.clone(),
        properties: edge.properties.clone(),
    }
}

fn graph_store_response_error(
    command: impl Into<String>,
    error: GraphStoreError,
    state_hash: ThgResult<String>,
) -> ThgResult<ThgResponse> {
    ThgResponse::try_err(
        command,
        ThgError::new(error.code, error.message),
        state_hash,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::{InMemoryThgExecutor, StoreBackedThgExecutor, ThgExecutor};
    use crate::commands::{ThgCommand, ThgRequest};
    use crate::errors::{ThgError, ThgResult};
    use crate::state::ThgState;
    use crate::store::ThgStore;

    #[derive(Clone, Debug)]
    struct FailingThgStore {
        load_error: bool,
    }

    impl ThgStore for FailingThgStore {
        fn load(&self) -> ThgResult<ThgState> {
            if self.load_error {
                Err(ThgError::new("store_unavailable", "load failed"))
            } else {
                Ok(ThgState::default())
            }
        }

        fn load_snapshot(&self) -> ThgResult<Arc<ThgState>> {
            self.load().map(Arc::new)
        }

        fn save(&mut self, _state: &ThgState) -> ThgResult<()> {
            Err(ThgError::new("persist_failed", "save failed"))
        }
    }

    #[test]
    fn command_sequence_updates_state_hash() -> ThgResult<()> {
        let mut executor = InMemoryThgExecutor::new();
        let first_hash = executor.state_hash()?;
        let begin = executor.execute_request(ThgRequest::new(
            ThgCommand::RunBegin.name(),
            json!({ "run_id": "run:1", "task": "ship THG" }),
        ))?;
        let step = executor.execute_request(ThgRequest::new(
            ThgCommand::RunStep.name(),
            json!({ "run_id": "run:1", "step_id": "step:1", "kind": "tool_call" }),
        ))?;

        assert!(begin.ok);
        assert!(step.ok);
        assert_ne!(first_hash, step.state_hash);
        assert_eq!(executor.state().runs["run:1"].steps.len(), 1);
        Ok(())
    }

    #[test]
    fn unknown_run_and_patch_mutations_fail_without_fabricated_success() -> ThgResult<()> {
        let mut executor = InMemoryThgExecutor::new();
        for (command, args, code) in [
            (
                ThgCommand::RunStep,
                json!({"run_id": "missing"}),
                "unknown_run_id",
            ),
            (
                ThgCommand::PatchValidate,
                json!({"patch_id": "missing"}),
                "unknown_patch_id",
            ),
            (
                ThgCommand::PatchCommit,
                json!({"patch_id": "missing"}),
                "unknown_patch_id",
            ),
        ] {
            let response = executor.execute_request(ThgRequest::new(command.name(), args))?;
            assert!(!response.ok);
            assert_eq!(response.status, code);
            assert!(response.nodes.is_empty());
            assert!(response.edges.is_empty());
            assert!(response.events.is_empty());
        }
        Ok(())
    }

    #[test]
    fn json_executor_returns_compatible_response_shape() -> ThgResult<()> {
        let mut executor = InMemoryThgExecutor::new();
        let raw = executor.execute_json(
            r#"{"command":"RUSTYRED_THG.CONTEXT.PACK","args":{"artifact_id":"artifact:1","sections":[]}}"#,
        )?;
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();

        assert_eq!(parsed["ok"], true);
        assert_eq!(parsed["command"], "RUSTYRED_THG.CONTEXT.PACK");
        assert_eq!(parsed["payload"]["artifact_id"], "artifact:1");
        assert!(parsed["state_hash"]
            .as_str()
            .unwrap()
            .starts_with("sha256:"));
        Ok(())
    }

    #[test]
    fn store_backed_executor_persists_after_mutating_command() -> ThgResult<()> {
        use crate::store::{InMemoryThgStore, ThgStore};

        let store = InMemoryThgStore::new();
        let mut executor = StoreBackedThgExecutor::new(store).unwrap();
        let response = executor.execute_request(ThgRequest::new(
            ThgCommand::RunBegin.name(),
            json!({ "run_id": "run:persisted", "task": "durable THG" }),
        ))?;

        assert!(response.ok);
        let saved = executor.store().load().unwrap();
        assert_eq!(saved.runs["run:persisted"].task, "durable THG");
        Ok(())
    }

    #[test]
    fn store_backed_executor_refuses_load_failures() {
        let error = StoreBackedThgExecutor::new(FailingThgStore { load_error: true }).unwrap_err();

        assert_eq!(error.code, "store_unavailable");
    }

    #[test]
    fn store_backed_executor_rolls_back_when_persist_fails() -> ThgResult<()> {
        let mut executor =
            StoreBackedThgExecutor::new(FailingThgStore { load_error: false }).unwrap();
        let initial_hash = executor.state_hash()?;

        let response = executor.execute_request(ThgRequest::new(
            ThgCommand::RunBegin.name(),
            json!({ "run_id": "run:not-persisted", "task": "must fail" }),
        ));

        let error = match response {
            Ok(response) => panic!("expected persistence failure, got {response:?}"),
            Err(error) => error,
        };
        assert_eq!(error.code, "persist_failed");
        assert_eq!(executor.state_hash()?, initial_hash);
        assert!(executor.state().runs.is_empty());
        Ok(())
    }

    #[test]
    fn stream_commands_publish_read_subscribe_and_ping() -> ThgResult<()> {
        use crate::store::{InMemoryThgStore, ThgStore};

        let store = InMemoryThgStore::new();
        let mut executor = StoreBackedThgExecutor::new(store).unwrap();

        // Subscribe bob (before any publish: he attends from "now").
        let sub = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamSubscribe.name(),
            json!({ "actor": "bob", "tenant": "acme", "stream": "room" }),
        ))?;
        assert!(sub.ok);
        let pre_hash = executor.state_hash()?;

        // Publish an info event; the state hash advances like any mutation.
        let publish = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamPublish.name(),
            json!({ "tenant": "acme", "stream": "room", "actor": "alice", "kind": "msg", "payload": { "n": 1 } }),
        ))?;
        assert!(publish.ok);
        assert_eq!(publish.payload["ordering_token"], 1);
        assert_ne!(
            pre_hash, publish.state_hash,
            "publish advances the state hash"
        );

        // bob reads exactly one event via his subscription; cursor advances.
        let read = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamRead.name(),
            json!({ "actor": "bob", "tenant": "acme", "advance": true }),
        ))?;
        assert!(read.ok);
        assert_eq!(read.payload["count"], 1);
        assert_eq!(read.events.len(), 1);

        // Re-read returns nothing (cursor consumed), and the cursor persisted.
        let reread = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamRead.name(),
            json!({ "actor": "bob", "tenant": "acme", "advance": true }),
        ))?;
        assert_eq!(reread.payload["count"], 0);
        assert_ne!(
            executor.store().load().unwrap().streams,
            crate::stream::StreamStore::default(),
            "stream state is persisted through the store-backed executor"
        );

        // A ping (ask + target) lands in the target's mention drain.
        let ping = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamPublish.name(),
            json!({ "tenant": "acme", "stream": "room", "actor": "alice", "kind": "q", "urgency": "ask", "target_actor": "carol" }),
        ))?;
        assert!(ping.ok);
        assert_eq!(ping.payload["pinged"], true);
        let mentions = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamMentions.name(),
            json!({ "actor": "carol", "tenant": "acme", "advance": true }),
        ))?;
        assert_eq!(mentions.payload["count"], 1);
        let mention_token = ping.payload["ordering_token"].as_u64().unwrap();
        let acknowledged = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamMentions.name(),
            json!({
                "actor": "carol",
                "tenant": "acme",
                "advance": false,
                "ack_ordering_tokens": [mention_token]
            }),
        ))?;
        assert_eq!(acknowledged.payload["acknowledged"], 1);
        assert_eq!(acknowledged.payload["count"], 0);

        let beta_ping = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamPublish.name(),
            json!({ "tenant": "beta", "stream": "room", "actor": "alice", "kind": "q", "urgency": "ask", "target_actor": "carol" }),
        ))?;
        assert!(beta_ping.ok);
        let wrong_tenant_mentions = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamMentions.name(),
            json!({ "actor": "carol", "tenant": "acme", "advance": false }),
        ))?;
        assert_eq!(
            wrong_tenant_mentions.payload["count"], 0,
            "tenant-scoped mention drain must not leak beta pings into acme"
        );
        let beta_mentions = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamMentions.name(),
            json!({ "actor": "carol", "tenant": "beta", "advance": true }),
        ))?;
        assert_eq!(beta_mentions.payload["count"], 1);

        // An ask without a target is rejected.
        let bad_ping = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamPublish.name(),
            json!({ "tenant": "acme", "stream": "room", "actor": "alice", "urgency": "ask" }),
        ))?;
        assert!(!bad_ping.ok);
        assert_eq!(
            bad_ping.error.as_ref().map(|e| e.code.as_str()),
            Some("missing_target_actor")
        );

        let missing_actor = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamPublish.name(),
            json!({ "tenant": "acme", "stream": "room" }),
        ))?;
        assert!(!missing_actor.ok);
        assert_eq!(
            missing_actor.error.as_ref().map(|e| e.code.as_str()),
            Some("missing_actor")
        );

        // Empty tenant is rejected, not routed to a default.
        let empty = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamPublish.name(),
            json!({ "tenant": "", "stream": "room", "actor": "alice" }),
        ))?;
        assert!(!empty.ok);
        assert_eq!(
            empty.error.as_ref().map(|e| e.code.as_str()),
            Some("empty_tenant")
        );
        Ok(())
    }

    #[test]
    fn stream_ack_command_stops_explicit_redelivery() -> ThgResult<()> {
        let mut executor = InMemoryThgExecutor::new();
        let publish = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamPublish.name(),
            json!({ "tenant": "acme", "stream": "room", "actor": "alice", "kind": "ask" }),
        ))?;
        assert!(publish.ok);

        let first = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamRead.name(),
            json!({
                "tenant": "acme",
                "actor": "bob",
                "streams": ["room"],
                "ack_policy": "explicit",
                "now_ms": 1000,
                "redeliver_after_ms": 100
            }),
        ))?;
        assert!(first.ok);
        assert_eq!(first.payload["count"], 1);
        assert_eq!(first.payload["pending"][0]["delivered_count"], 1);

        let redelivered = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamRead.name(),
            json!({
                "tenant": "acme",
                "actor": "bob",
                "streams": ["room"],
                "ack_policy": "explicit",
                "now_ms": 1200,
                "redeliver_after_ms": 100
            }),
        ))?;
        assert_eq!(redelivered.payload["count"], 1);
        assert_eq!(redelivered.payload["pending"][0]["delivered_count"], 2);

        let ack = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamAck.name(),
            json!({
                "tenant": "acme",
                "actor": "bob",
                "stream": "room",
                "ordering_tokens": [1]
            }),
        ))?;
        assert!(ack.ok);
        assert_eq!(ack.payload["acked"], 1);

        let after_ack = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamRead.name(),
            json!({
                "tenant": "acme",
                "actor": "bob",
                "streams": ["room"],
                "ack_policy": "explicit",
                "now_ms": 1400,
                "redeliver_after_ms": 100
            }),
        ))?;
        assert_eq!(after_ack.payload["count"], 0);
        assert_eq!(after_ack.payload["dead_lettered"], json!([]));
        Ok(())
    }

    #[test]
    fn stream_read_explicit_advance_false_peeks_without_delivery() -> ThgResult<()> {
        let mut executor = InMemoryThgExecutor::new();
        let publish = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamPublish.name(),
            json!({ "tenant": "acme", "stream": "room", "actor": "alice", "kind": "ask" }),
        ))?;
        assert!(publish.ok);

        let peek = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamRead.name(),
            json!({
                "tenant": "acme",
                "actor": "bob",
                "streams": ["room"],
                "ack_policy": "explicit",
                "advance": false,
                "now_ms": 1000,
                "redeliver_after_ms": 100
            }),
        ))?;
        assert!(peek.ok);
        assert_eq!(peek.payload["count"], 1);
        assert_eq!(peek.payload["advanced"], false);
        assert_eq!(peek.payload["ack_policy"], "explicit");
        assert_eq!(peek.payload["pending"], json!([]));
        assert_eq!(peek.payload["dead_lettered"], json!([]));

        let repeat_peek = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamRead.name(),
            json!({
                "tenant": "acme",
                "actor": "bob",
                "streams": ["room"],
                "ack_policy": "explicit",
                "advance": false,
                "now_ms": 1001,
                "redeliver_after_ms": 100
            }),
        ))?;
        assert!(repeat_peek.ok);
        assert_eq!(repeat_peek.payload["count"], 1);
        assert_eq!(repeat_peek.payload["pending"], json!([]));

        let delivered = executor.execute_request(ThgRequest::new(
            ThgCommand::StreamRead.name(),
            json!({
                "tenant": "acme",
                "actor": "bob",
                "streams": ["room"],
                "ack_policy": "explicit",
                "advance": true,
                "now_ms": 1002,
                "redeliver_after_ms": 100
            }),
        ))?;
        assert!(delivered.ok);
        assert_eq!(delivered.payload["count"], 1);
        assert_eq!(delivered.payload["advanced"], true);
        assert_eq!(delivered.payload["pending"][0]["delivered_count"], 1);
        Ok(())
    }

    #[test]
    fn graph_commands_upsert_query_and_verify_graph_store() -> ThgResult<()> {
        let mut executor = InMemoryThgExecutor::new();
        let node_a = executor.execute_request(ThgRequest::new(
            ThgCommand::GraphNodeUpsert.name(),
            json!({
                "id": "node:a",
                "labels": ["File"],
                "properties": { "path": "src/lib.rs", "repo": "rusty-red" }
            }),
        ))?;
        let node_b = executor.execute_request(ThgRequest::new(
            ThgCommand::GraphNodeUpsert.name(),
            json!({
                "id": "node:b",
                "labels": ["File"],
                "properties": { "path": "src/main.rs", "repo": "rusty-red" }
            }),
        ))?;
        let edge = executor.execute_request(ThgRequest::new(
            ThgCommand::GraphEdgeUpsert.name(),
            json!({
                "id": "edge:ab",
                "from_id": "node:a",
                "type": "IMPORTS",
                "to_id": "node:b",
                "properties": { "weight": 1 }
            }),
        ))?;
        let query = executor.execute_request(ThgRequest::new(
            ThgCommand::GraphNodesQuery.name(),
            json!({
                "label": "File",
                "properties": { "path": "src/lib.rs" }
            }),
        ))?;
        let neighbors = executor.execute_request(ThgRequest::new(
            ThgCommand::GraphNeighbors.name(),
            json!({ "node_id": "node:a", "direction": "out" }),
        ))?;
        let verify =
            executor.execute_request(ThgRequest::new(ThgCommand::GraphVerify.name(), json!({})))?;
        let rebuild = executor.execute_request(ThgRequest::new(
            ThgCommand::GraphRebuildIndexes.name(),
            json!({}),
        ))?;

        assert!(node_a.ok);
        assert!(node_b.ok);
        assert!(edge.ok);
        assert_eq!(query.payload["plan"]["operation"], "node_index_seek");
        assert_eq!(query.payload["stats"]["returned"], 1);
        assert_eq!(query.nodes[0].id, "node:a");
        assert_eq!(neighbors.payload["plan"]["operation"], "adjacency_seek");
        assert_eq!(neighbors.payload["neighbors"][0]["node_id"], "node:b");
        assert_eq!(verify.payload["report"]["ok"], true);
        assert_eq!(rebuild.payload["report"]["after"]["ok"], true);
        Ok(())
    }

    #[test]
    fn graph_edge_command_requires_live_endpoints() -> ThgResult<()> {
        let mut executor = InMemoryThgExecutor::new();
        executor.execute_request(ThgRequest::new(
            ThgCommand::GraphNodeUpsert.name(),
            json!({ "id": "node:a", "labels": ["File"] }),
        ))?;

        let response = executor.execute_request(ThgRequest::new(
            ThgCommand::GraphEdgeUpsert.name(),
            json!({
                "id": "edge:missing",
                "from_id": "node:a",
                "type": "IMPORTS",
                "to_id": "node:missing"
            }),
        ))?;

        assert!(!response.ok);
        assert_eq!(
            response.error.as_ref().map(|error| error.code.as_str()),
            Some("missing_graph_endpoint")
        );
        Ok(())
    }
}
