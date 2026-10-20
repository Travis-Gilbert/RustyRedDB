use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::access_method::AccessMethod;
use crate::graph::error::{GraphStoreError, GraphStoreResult};
use crate::graph_store::RedCoreGraphStore;
use crate::hooks::HookRegistration;

pub type PluginOperationHandler = fn(PluginOperationContext<'_>, Value) -> GraphStoreResult<Value>;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginCapabilityKind {
    Designation,
    Encoder,
    Index,
    Operation,
    Hook,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PluginCapability {
    pub kind: PluginCapabilityKind,
    pub name: String,
}

#[derive(Clone)]
pub struct PluginOperationRegistration {
    pub operation: &'static str,
    pub command: &'static str,
    pub aliases: &'static [&'static str],
    pub summary: &'static str,
    pub writes_graph: bool,
    pub handler: PluginOperationHandler,
}

impl std::fmt::Debug for PluginOperationRegistration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginOperationRegistration")
            .field("operation", &self.operation)
            .field("command", &self.command)
            .field("aliases", &self.aliases)
            .field("summary", &self.summary)
            .field("writes_graph", &self.writes_graph)
            .finish_non_exhaustive()
    }
}

pub struct PluginOperationContext<'a> {
    pub tenant_id: &'a str,
    pub operation: &'a str,
    pub command: &'a str,
    pub store: &'a mut RedCoreGraphStore,
}

pub trait RustyRedPlugin: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;

    fn capabilities(&self) -> Vec<PluginCapability> {
        Vec::new()
    }

    fn operations(&self) -> Vec<PluginOperationRegistration> {
        Vec::new()
    }

    /// Planner-facing access methods supplied by index-capability plugins.
    /// Command dispatch remains available for direct operations; this seam lets
    /// the relational planner ask loaded indexes for cost and row-id scans.
    fn access_methods(&self) -> Vec<Arc<dyn AccessMethod>> {
        Vec::new()
    }

    /// Graph-level hooks this plugin registers. Default empty so existing
    /// plugins compile unchanged. The registry collects these at init; an
    /// embedder feeds them to a `crate::hooks::HookDispatcher`.
    fn hooks(&self) -> Vec<HookRegistration> {
        Vec::new()
    }
}

#[derive(Clone, Default)]
pub struct PluginRegistry {
    plugins: Vec<Arc<dyn RustyRedPlugin>>,
    operations: BTreeMap<String, PluginOperationRegistration>,
    access_methods: Vec<Arc<dyn AccessMethod>>,
}

impl std::fmt::Debug for PluginRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginRegistry")
            .field("plugins", &self.plugins.len())
            .field("operations", &self.operations)
            .field(
                "access_methods",
                &self
                    .access_methods
                    .iter()
                    .map(|method| method.name())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, plugin: impl RustyRedPlugin + 'static) {
        let plugin = Arc::new(plugin);
        for operation in plugin.operations() {
            self.insert_operation(operation);
        }
        self.access_methods.extend(plugin.access_methods());
        self.plugins.push(plugin);
    }

    pub fn plugins(&self) -> Vec<&dyn RustyRedPlugin> {
        self.plugins
            .iter()
            .map(|plugin| plugin.as_ref() as &dyn RustyRedPlugin)
            .collect()
    }

    pub fn capabilities(&self) -> Vec<PluginCapability> {
        self.plugins
            .iter()
            .flat_map(|plugin| plugin.capabilities())
            .collect()
    }

    /// All graph-level hooks registered by loaded plugins, collected at the
    /// point of call. An embedder feeds these to a `crate::hooks::HookDispatcher`
    /// (typically once at init) so plugin hooks fire on store mutations.
    pub fn hooks(&self) -> Vec<HookRegistration> {
        self.plugins
            .iter()
            .flat_map(|plugin| plugin.hooks())
            .collect()
    }

    pub fn operation(&self, command: &str) -> Option<&PluginOperationRegistration> {
        self.operations.get(&normalize_plugin_command(command))
    }

    pub fn operations(&self) -> Vec<&PluginOperationRegistration> {
        self.operations.values().collect()
    }

    pub fn access_methods(&self) -> Vec<&dyn AccessMethod> {
        self.access_methods
            .iter()
            .map(|method| method.as_ref() as &dyn AccessMethod)
            .collect()
    }

    pub fn execute(
        &self,
        store: &mut RedCoreGraphStore,
        tenant_id: &str,
        command: &str,
        arguments: Value,
    ) -> GraphStoreResult<PluginExecutionOutput> {
        let registration = self.operation(command).cloned().ok_or_else(|| {
            GraphStoreError::new(
                "unknown_plugin_operation",
                format!("unknown plugin operation: {command}"),
            )
        })?;

        if registration.writes_graph && plugin_arg_bool(&arguments, &["dry_run", "dryRun"]) {
            return Ok(PluginExecutionOutput {
                tenant_id: tenant_id.to_string(),
                operation: registration.operation.to_string(),
                command: registration.command.to_string(),
                writes_graph: registration.writes_graph,
                result: json!({
                    "status": "dry_run",
                    "message": "dry_run=true: no graph writes committed",
                    "writes_graph": true,
                }),
            });
        }

        let context = PluginOperationContext {
            tenant_id,
            operation: registration.operation,
            command: registration.command,
            store,
        };
        let result = (registration.handler)(context, arguments)?;
        Ok(PluginExecutionOutput {
            tenant_id: tenant_id.to_string(),
            operation: registration.operation.to_string(),
            command: registration.command.to_string(),
            writes_graph: registration.writes_graph,
            result,
        })
    }

    fn insert_operation(&mut self, operation: PluginOperationRegistration) {
        self.operations.insert(
            normalize_plugin_command(operation.command),
            operation.clone(),
        );
        self.operations.insert(
            normalize_plugin_command(operation.operation),
            operation.clone(),
        );
        for alias in operation.aliases {
            self.operations
                .insert(normalize_plugin_command(alias), operation.clone());
        }
    }
}

#[derive(Clone, Debug)]
pub struct PluginExecutionOutput {
    pub tenant_id: String,
    pub operation: String,
    pub command: String,
    pub writes_graph: bool,
    pub result: Value,
}

impl PluginExecutionOutput {
    pub fn to_json(&self) -> Value {
        json!({
            "tenant_id": self.tenant_id,
            "operation": self.operation,
            "command": self.command,
            "writes_graph": self.writes_graph,
            "result": self.result,
        })
    }
}

pub fn normalize_plugin_command(command: &str) -> String {
    command.trim().to_ascii_uppercase()
}

fn plugin_arg_bool(arguments: &Value, keys: &[&str]) -> bool {
    keys.iter()
        .find_map(|key| arguments.get(*key).and_then(Value::as_bool))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_method::OrderedAccessMethod;

    #[derive(Debug)]
    struct NoopPlugin;

    impl RustyRedPlugin for NoopPlugin {
        fn name(&self) -> &'static str {
            "test.noop"
        }

        fn capabilities(&self) -> Vec<PluginCapability> {
            vec![PluginCapability {
                kind: PluginCapabilityKind::Operation,
                name: "noop.echo".to_string(),
            }]
        }

        fn operations(&self) -> Vec<PluginOperationRegistration> {
            vec![PluginOperationRegistration {
                operation: "echo",
                command: "RUSTYRED_THG.PLUGIN.ECHO",
                aliases: &["plugin.echo", "test.noop.echo"],
                summary: "Echo the JSON payload.",
                writes_graph: false,
                handler: |context, arguments| {
                    Ok(json!({
                        "tenant_id": context.tenant_id,
                        "operation": context.operation,
                        "command": context.command,
                        "arguments": arguments,
                    }))
                },
            }]
        }
    }

    #[test]
    fn registry_resolves_plugin_operation_aliases() {
        let mut registry = PluginRegistry::new();
        registry.register(NoopPlugin);

        assert_eq!(registry.plugins()[0].name(), "test.noop");
        assert_eq!(registry.capabilities()[0].name, "noop.echo");
        assert!(registry.operation("plugin.echo").is_some());
        assert!(registry.operation("RUSTYRED_THG.PLUGIN.ECHO").is_some());
        assert!(registry.operation("test.noop.echo").is_some());
    }

    #[test]
    fn registry_executes_with_store_context() {
        let mut registry = PluginRegistry::new();
        registry.register(NoopPlugin);
        let mut store = RedCoreGraphStore::memory();

        let output = registry
            .execute(&mut store, "tenant-a", "plugin.echo", json!({ "value": 1 }))
            .unwrap();

        assert_eq!(output.tenant_id, "tenant-a");
        assert_eq!(output.operation, "echo");
        assert_eq!(output.command, "RUSTYRED_THG.PLUGIN.ECHO");
        assert!(!output.writes_graph);
        assert_eq!(output.result["arguments"]["value"], 1);
    }

    #[derive(Debug)]
    struct IndexPlugin;

    impl RustyRedPlugin for IndexPlugin {
        fn name(&self) -> &'static str {
            "test.index"
        }

        fn capabilities(&self) -> Vec<PluginCapability> {
            vec![PluginCapability {
                kind: PluginCapabilityKind::Index,
                name: "ordered".to_string(),
            }]
        }

        fn access_methods(&self) -> Vec<Arc<dyn AccessMethod>> {
            vec![Arc::new(OrderedAccessMethod::new())]
        }
    }

    #[test]
    fn registry_exposes_index_plugin_access_methods() {
        let mut registry = PluginRegistry::new();
        registry.register(IndexPlugin);

        assert_eq!(registry.capabilities()[0].kind, PluginCapabilityKind::Index);
        assert_eq!(registry.access_methods()[0].name(), "ordered");
    }
}
