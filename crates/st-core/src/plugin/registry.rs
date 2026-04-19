// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Plugin registry for managing loaded plugins.

use super::context::PluginContext;
use super::events::{PluginEvent, PluginEventBus};
use super::traits::{Plugin, PluginCapability, PluginMetadata};
use crate::{PureResult, TensorError};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};

/// Summary of dependency issues among currently registered plugins.
#[derive(Debug, Clone)]
pub struct DependencyValidationSummary {
    /// True when no missing dependencies or dependency cycles are detected.
    pub ok: bool,
    /// Plugins whose dependencies are not currently registered.
    pub missing: HashMap<String, Vec<String>>,
    /// Detected dependency cycles (each cycle is a list of plugin IDs).
    pub cycles: Vec<Vec<String>>,
}

/// Handle to a loaded plugin.
#[derive(Clone)]
pub struct PluginHandle {
    plugin: Arc<Mutex<Box<dyn Plugin>>>,
}

impl PluginHandle {
    fn new(plugin: Box<dyn Plugin>) -> Self {
        Self {
            plugin: Arc::new(Mutex::new(plugin)),
        }
    }

    /// Get the plugin's metadata.
    pub fn metadata(&self) -> PluginMetadata {
        self.plugin.lock().unwrap().metadata()
    }

    /// Execute a function with access to the plugin.
    pub fn with_plugin<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut dyn Plugin) -> R,
    {
        let mut plugin = self.plugin.lock().unwrap();
        f(&mut **plugin)
    }
}

/// Registry managing all loaded plugins.
pub struct PluginRegistry {
    plugins: RwLock<HashMap<String, PluginHandle>>,
    context: Arc<Mutex<PluginContext>>,
    event_bus: PluginEventBus,
}

impl PluginRegistry {
    /// Create a new plugin registry.
    pub fn new() -> Self {
        let event_bus = PluginEventBus::new();
        let context = PluginContext::new(event_bus.clone());

        Self {
            plugins: RwLock::new(HashMap::new()),
            context: Arc::new(Mutex::new(context)),
            event_bus,
        }
    }

    /// Register a new plugin.
    ///
    /// This loads the plugin, validates dependencies, and calls its `on_load` hook.
    pub fn register(&self, mut plugin: Box<dyn Plugin>) -> PureResult<()> {
        let metadata = plugin.metadata();
        let plugin_id = metadata.id.clone();

        // Check if already registered
        if self.plugins.read().unwrap().contains_key(&plugin_id) {
            return Err(TensorError::Generic(format!(
                "Plugin '{}' is already registered",
                plugin_id
            )));
        }

        // Validate dependencies
        self.validate_dependencies(&metadata)?;

        // Call on_load hook (avoid holding the registry context lock while executing plugin code)
        let mut ctx = { self.context.lock().unwrap().clone() };
        plugin.on_load(&mut ctx)?;

        // Store the plugin
        let handle = PluginHandle::new(plugin);
        self.plugins
            .write()
            .unwrap()
            .insert(plugin_id.clone(), handle);

        // Emit event
        self.event_bus
            .publish(&PluginEvent::PluginLoaded { plugin_id });

        Ok(())
    }

    /// Unregister a plugin by ID.
    pub fn unregister(&self, plugin_id: &str) -> PureResult<()> {
        let handle = {
            let mut plugins = self.plugins.write().unwrap();
            plugins
                .remove(plugin_id)
                .ok_or_else(|| TensorError::Generic(format!("Plugin '{}' not found", plugin_id)))?
        };

        // Call on_unload hook
        let mut ctx = { self.context.lock().unwrap().clone() };
        handle.with_plugin(|plugin: &mut dyn Plugin| plugin.on_unload(&mut ctx))?;

        // Emit event
        self.event_bus.publish(&PluginEvent::PluginUnloaded {
            plugin_id: plugin_id.to_string(),
        });

        Ok(())
    }

    /// Get a handle to a registered plugin.
    pub fn get(&self, plugin_id: &str) -> Option<PluginHandle> {
        self.plugins.read().unwrap().get(plugin_id).cloned()
    }

    /// Find plugins by capability.
    pub fn find_by_capability(&self, capability: &PluginCapability) -> Vec<PluginHandle> {
        self.plugins
            .read()
            .unwrap()
            .values()
            .filter(|handle| {
                let meta = handle.metadata();
                meta.capabilities.contains(capability)
            })
            .cloned()
            .collect()
    }

    /// List all registered plugin IDs.
    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins.read().unwrap().keys().cloned().collect()
    }

    /// Return a dependency adjacency list for the currently registered plugins.
    ///
    /// When `internal_only` is true, dependencies that are not registered are omitted.
    pub fn dependency_graph(&self, internal_only: bool) -> HashMap<String, Vec<String>> {
        let plugins = self.plugins.read().unwrap();
        let mut ids: Vec<String> = plugins.keys().cloned().collect();
        ids.sort();

        let id_set: HashSet<&str> = ids.iter().map(|id| id.as_str()).collect();
        let mut graph = HashMap::with_capacity(ids.len());

        for id in &ids {
            let Some(handle) = plugins.get(id) else {
                graph.insert(id.clone(), Vec::new());
                continue;
            };
            let meta = handle.metadata();
            let mut deps: Vec<String> = meta.dependencies.keys().cloned().collect();
            if internal_only {
                deps.retain(|dep| id_set.contains(dep.as_str()));
            }
            deps.sort();
            deps.dedup();
            graph.insert(id.clone(), deps);
        }

        graph
    }

    /// Validate dependency metadata for missing dependencies and dependency cycles.
    ///
    /// Missing dependencies can arise if a plugin is unregistered while dependents remain loaded.
    /// Dependency cycles can arise via hot-reload/replace flows that update plugin metadata.
    pub fn validate_dependency_graph(&self, internal_only: bool) -> DependencyValidationSummary {
        let graph = self.dependency_graph(false);
        let mut plugin_ids: Vec<String> = graph.keys().cloned().collect();
        plugin_ids.sort();

        let plugin_set: HashSet<&str> = plugin_ids.iter().map(|id| id.as_str()).collect();

        let mut missing: HashMap<String, Vec<String>> = HashMap::new();
        let mut internal_graph: HashMap<String, Vec<String>> =
            HashMap::with_capacity(plugin_ids.len());

        for plugin_id in &plugin_ids {
            let deps = graph.get(plugin_id).cloned().unwrap_or_default();
            let mut internal_deps = Vec::new();
            let mut missing_deps = Vec::new();

            for dep in deps {
                if plugin_set.contains(dep.as_str()) {
                    internal_deps.push(dep);
                } else if !internal_only {
                    missing_deps.push(dep);
                }
            }

            internal_deps.sort();
            internal_deps.dedup();
            missing_deps.sort();
            missing_deps.dedup();

            internal_graph.insert(plugin_id.clone(), internal_deps);
            if !internal_only && !missing_deps.is_empty() {
                missing.insert(plugin_id.clone(), missing_deps);
            }
        }

        let mut cycles = detect_dependency_cycles(&plugin_ids, &internal_graph);
        cycles.sort_by(|left, right| left.len().cmp(&right.len()).then(left.cmp(right)));

        let ok = missing.is_empty() && cycles.is_empty();
        DependencyValidationSummary {
            ok,
            missing,
            cycles,
        }
    }

    /// Unregister a plugin and any currently registered plugins that depend on it.
    ///
    /// This prevents leaving dependents registered with missing dependencies.
    pub fn unregister_safe(&self, plugin_id: &str, strict: bool) -> PureResult<Vec<String>> {
        let plugin_id = plugin_id.trim();
        if plugin_id.is_empty() {
            return Err(TensorError::Generic(
                "plugin_id must not be empty".to_string(),
            ));
        }

        if self.get(plugin_id).is_none() {
            if strict {
                return Err(TensorError::Generic(format!(
                    "Plugin '{}' not found",
                    plugin_id
                )));
            }
            return Ok(Vec::new());
        }

        let graph = self.dependency_graph(false);
        let mut dependents: HashMap<String, Vec<String>> = HashMap::new();
        for (pid, deps) in &graph {
            for dep in deps {
                dependents.entry(dep.clone()).or_default().push(pid.clone());
            }
        }
        for bucket in dependents.values_mut() {
            bucket.sort();
            bucket.dedup();
        }

        let mut match_set: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(plugin_id.to_string());
        while let Some(current) = queue.pop_front() {
            if !match_set.insert(current.clone()) {
                continue;
            }
            if let Some(children) = dependents.get(&current) {
                for child in children {
                    if !match_set.contains(child) {
                        queue.push_back(child.clone());
                    }
                }
            }
        }

        let mut matches: Vec<String> = match_set.iter().cloned().collect();
        matches.sort();
        if matches.is_empty() {
            return Ok(Vec::new());
        }

        let mut indegree: HashMap<String, usize> =
            matches.iter().map(|id| (id.clone(), 0)).collect();
        let mut edges: HashMap<String, Vec<String>> =
            matches.iter().map(|id| (id.clone(), Vec::new())).collect();

        for pid in &matches {
            let deps = graph.get(pid).map(|deps| deps.as_slice()).unwrap_or(&[]);
            for dep in deps {
                if !match_set.contains(dep) {
                    continue;
                }
                edges.entry(dep.clone()).or_default().push(pid.clone());
                *indegree.entry(pid.clone()).or_insert(0) += 1;
            }
        }

        let mut topo_queue = VecDeque::new();
        for id in &matches {
            if indegree.get(id).copied().unwrap_or(0) == 0 {
                topo_queue.push_back(id.clone());
            }
        }

        let mut order = Vec::with_capacity(matches.len());
        while let Some(id) = topo_queue.pop_front() {
            order.push(id.clone());
            let Some(children) = edges.get(&id) else {
                continue;
            };
            for child in children {
                let Some(entry) = indegree.get_mut(child) else {
                    continue;
                };
                *entry = entry.saturating_sub(1);
                if *entry == 0 {
                    topo_queue.push_back(child.clone());
                }
            }
        }

        let unload_order: Vec<String> = if order.len() == matches.len() {
            order.into_iter().rev().collect()
        } else {
            let mut ids = matches;
            ids.reverse();
            ids
        };

        let mut unloaded = Vec::with_capacity(unload_order.len());
        for pid in unload_order {
            self.unregister(&pid)?;
            unloaded.push(pid);
        }

        Ok(unloaded)
    }

    /// Initialize all registered plugins.
    ///
    /// This is typically called after all plugins have been registered.
    pub fn initialize_all(&self) -> PureResult<()> {
        self.event_bus.publish(&PluginEvent::SystemInit);
        Ok(())
    }

    /// Shutdown all plugins.
    pub fn shutdown(&self) -> PureResult<()> {
        self.event_bus.publish(&PluginEvent::SystemShutdown);

        let (plugin_ids, deps_by_id) = {
            let plugins = self.plugins.read().unwrap();
            let mut ids: Vec<String> = plugins.keys().cloned().collect();
            ids.sort();

            let mut deps_by_id = HashMap::new();
            for id in &ids {
                let Some(handle) = plugins.get(id) else {
                    deps_by_id.insert(id.clone(), Vec::new());
                    continue;
                };
                let meta = handle.metadata();
                let mut deps: Vec<String> = meta
                    .dependencies
                    .keys()
                    .filter(|dep| plugins.contains_key(*dep))
                    .cloned()
                    .collect();
                deps.sort();
                deps_by_id.insert(id.clone(), deps);
            }
            (ids, deps_by_id)
        };

        if plugin_ids.is_empty() {
            return Ok(());
        }

        let plugin_id_set: std::collections::HashSet<&str> =
            plugin_ids.iter().map(|id| id.as_str()).collect();
        let mut indegree: HashMap<String, usize> =
            plugin_ids.iter().map(|id| (id.clone(), 0)).collect();
        let mut edges: HashMap<String, Vec<String>> = plugin_ids
            .iter()
            .map(|id| (id.clone(), Vec::new()))
            .collect();

        for id in &plugin_ids {
            let deps = deps_by_id
                .get(id)
                .map(|deps| deps.as_slice())
                .unwrap_or(&[]);
            for dep in deps {
                if !plugin_id_set.contains(dep.as_str()) {
                    continue;
                }
                edges.entry(dep.clone()).or_default().push(id.clone());
                *indegree.entry(id.clone()).or_insert(0) += 1;
            }
        }

        let mut queue = VecDeque::new();
        for id in &plugin_ids {
            if indegree.get(id).copied().unwrap_or(0) == 0 {
                queue.push_back(id.clone());
            }
        }

        let mut order = Vec::with_capacity(plugin_ids.len());
        while let Some(id) = queue.pop_front() {
            order.push(id.clone());
            let Some(children) = edges.get(&id) else {
                continue;
            };
            for child in children {
                let Some(entry) = indegree.get_mut(child) else {
                    continue;
                };
                *entry = entry.saturating_sub(1);
                if *entry == 0 {
                    queue.push_back(child.clone());
                }
            }
        }

        let unload_order: Vec<String> = if order.len() == plugin_ids.len() {
            order.into_iter().rev().collect()
        } else {
            // Cycles shouldn't be possible given the registry enforces dependencies at registration time,
            // but fall back to a deterministic order to avoid leaving the system partially shut down.
            let mut ids = plugin_ids;
            ids.reverse();
            ids
        };

        for plugin_id in unload_order {
            self.unregister(&plugin_id)?;
        }

        Ok(())
    }

    /// Get the event bus.
    pub fn event_bus(&self) -> &PluginEventBus {
        &self.event_bus
    }

    /// Get the plugin context.
    pub fn context(&self) -> Arc<Mutex<PluginContext>> {
        Arc::clone(&self.context)
    }

    fn validate_dependencies(&self, metadata: &PluginMetadata) -> PureResult<()> {
        for dep_id in metadata.dependencies.keys() {
            if !self.plugins.read().unwrap().contains_key(dep_id) {
                return Err(TensorError::Generic(format!(
                    "Plugin '{}' depends on '{}' which is not registered",
                    metadata.id, dep_id
                )));
            }
        }
        Ok(())
    }
}

fn canonical_cycle(nodes: &[String]) -> Vec<String> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let mut best: Vec<String> = Vec::new();
    for idx in 0..nodes.len() {
        let mut rotated = Vec::with_capacity(nodes.len());
        rotated.extend_from_slice(&nodes[idx..]);
        rotated.extend_from_slice(&nodes[..idx]);
        if best.is_empty() || rotated < best {
            best = rotated;
        }
    }
    best
}

fn detect_dependency_cycles(
    plugin_ids: &[String],
    graph: &HashMap<String, Vec<String>>,
) -> Vec<Vec<String>> {
    let mut id_to_idx = HashMap::with_capacity(plugin_ids.len());
    for (idx, id) in plugin_ids.iter().enumerate() {
        id_to_idx.insert(id.clone(), idx);
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); plugin_ids.len()];
    for (id, deps) in graph {
        let Some(&src) = id_to_idx.get(id) else {
            continue;
        };
        let mut out = Vec::new();
        for dep in deps {
            if let Some(&dst) = id_to_idx.get(dep) {
                out.push(dst);
            }
        }
        out.sort();
        out.dedup();
        adj[src] = out;
    }

    let mut state = vec![0u8; plugin_ids.len()];
    let mut stack: Vec<usize> = Vec::new();
    let mut positions: Vec<Option<usize>> = vec![None; plugin_ids.len()];
    let mut cycle_keys: HashSet<String> = HashSet::new();
    let mut cycles: Vec<Vec<String>> = Vec::new();

    fn dfs(
        node: usize,
        adj: &[Vec<usize>],
        plugin_ids: &[String],
        state: &mut [u8],
        stack: &mut Vec<usize>,
        positions: &mut [Option<usize>],
        cycle_keys: &mut HashSet<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        state[node] = 1;
        positions[node] = Some(stack.len());
        stack.push(node);

        for &dep in &adj[node] {
            match state.get(dep).copied().unwrap_or(2) {
                0 => dfs(
                    dep, adj, plugin_ids, state, stack, positions, cycle_keys, cycles,
                ),
                1 => {
                    if let Some(start) = positions.get(dep).and_then(|v| *v) {
                        let slice = &stack[start..];
                        let nodes: Vec<String> =
                            slice.iter().map(|idx| plugin_ids[*idx].clone()).collect();
                        let canon = canonical_cycle(&nodes);
                        if !canon.is_empty() {
                            let key = canon.join("\u{1f}");
                            if cycle_keys.insert(key) {
                                cycles.push(canon);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        stack.pop();
        positions[node] = None;
        state[node] = 2;
    }

    for idx in 0..plugin_ids.len() {
        if state[idx] == 0 {
            dfs(
                idx,
                &adj,
                plugin_ids,
                &mut state,
                &mut stack,
                &mut positions,
                &mut cycle_keys,
                &mut cycles,
            );
        }
    }

    cycles
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugin::traits::{Plugin, PluginMetadata};
    use std::any::Any;
    use std::sync::{Arc, Mutex};

    struct TestPlugin {
        name: String,
    }

    impl Plugin for TestPlugin {
        fn metadata(&self) -> PluginMetadata {
            PluginMetadata::new(&self.name, "1.0.0").with_capability(PluginCapability::Operators)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn test_plugin_registration() {
        let registry = PluginRegistry::new();
        let plugin = Box::new(TestPlugin {
            name: "test".to_string(),
        });

        assert!(registry.register(plugin).is_ok());
        assert!(registry.get("test").is_some());
    }

    #[test]
    fn test_plugin_unregistration() {
        let registry = PluginRegistry::new();
        let plugin = Box::new(TestPlugin {
            name: "test".to_string(),
        });

        registry.register(plugin).unwrap();
        assert!(registry.unregister("test").is_ok());
        assert!(registry.get("test").is_none());
    }

    #[test]
    fn test_find_by_capability() {
        let registry = PluginRegistry::new();

        registry
            .register(Box::new(TestPlugin {
                name: "plugin1".to_string(),
            }))
            .unwrap();

        let plugins = registry.find_by_capability(&PluginCapability::Operators);
        assert_eq!(plugins.len(), 1);
    }

    #[test]
    fn test_shutdown_dependency_order() {
        struct DepPlugin {
            id: String,
            deps: Vec<String>,
            unload_log: Arc<Mutex<Vec<String>>>,
        }

        impl Plugin for DepPlugin {
            fn metadata(&self) -> PluginMetadata {
                let mut meta = PluginMetadata::new(&self.id, "1.0.0");
                for dep in &self.deps {
                    meta = meta.with_dependency(dep.clone(), ">=0");
                }
                meta
            }

            fn on_unload(&mut self, _ctx: &mut PluginContext) -> PureResult<()> {
                self.unload_log.lock().unwrap().push(self.id.clone());
                Ok(())
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }

        let registry = PluginRegistry::new();
        let unload_log = Arc::new(Mutex::new(Vec::new()));

        registry
            .register(Box::new(DepPlugin {
                id: "a".to_string(),
                deps: Vec::new(),
                unload_log: unload_log.clone(),
            }))
            .unwrap();
        registry
            .register(Box::new(DepPlugin {
                id: "b".to_string(),
                deps: vec!["a".to_string()],
                unload_log: unload_log.clone(),
            }))
            .unwrap();
        registry
            .register(Box::new(DepPlugin {
                id: "c".to_string(),
                deps: Vec::new(),
                unload_log: unload_log.clone(),
            }))
            .unwrap();

        registry.shutdown().unwrap();
        let unloaded = unload_log.lock().unwrap().clone();
        assert_eq!(
            unloaded,
            vec!["b".to_string(), "c".to_string(), "a".to_string()]
        );
        assert!(registry.list_plugins().is_empty());
    }

    #[test]
    fn test_dependency_graph_and_validation() {
        struct DepPlugin {
            id: String,
            deps: Vec<String>,
        }

        impl Plugin for DepPlugin {
            fn metadata(&self) -> PluginMetadata {
                let mut meta = PluginMetadata::new(&self.id, "1.0.0");
                for dep in &self.deps {
                    meta = meta.with_dependency(dep.clone(), ">=0");
                }
                meta
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }

        let registry = PluginRegistry::new();
        registry
            .register(Box::new(DepPlugin {
                id: "a".to_string(),
                deps: Vec::new(),
            }))
            .unwrap();
        registry
            .register(Box::new(DepPlugin {
                id: "b".to_string(),
                deps: vec!["a".to_string()],
            }))
            .unwrap();

        let graph = registry.dependency_graph(true);
        assert_eq!(
            graph.get("a").cloned().unwrap_or_default(),
            Vec::<String>::new()
        );
        assert_eq!(
            graph.get("b").cloned().unwrap_or_default(),
            vec!["a".to_string()]
        );

        registry.unregister("a").unwrap();
        let summary = registry.validate_dependency_graph(false);
        assert!(!summary.ok);
        assert_eq!(
            summary.missing.get("b").cloned().unwrap_or_default(),
            vec!["a".to_string()]
        );

        let summary_internal = registry.validate_dependency_graph(true);
        assert!(summary_internal.ok);
        assert!(summary_internal.missing.is_empty());
        assert!(summary_internal.cycles.is_empty());

        registry
            .register(Box::new(DepPlugin {
                id: "a".to_string(),
                deps: vec!["b".to_string()],
            }))
            .unwrap();
        let summary_cycle = registry.validate_dependency_graph(false);
        assert!(!summary_cycle.ok);
        assert!(summary_cycle.missing.is_empty());
        assert!(summary_cycle.cycles.iter().any(|cycle| cycle
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>(
        ) == std::collections::HashSet::from([
            "a".to_string(),
            "b".to_string()
        ])));
    }

    #[test]
    fn test_unregister_safe() {
        struct DepPlugin {
            id: String,
            deps: Vec<String>,
        }

        impl Plugin for DepPlugin {
            fn metadata(&self) -> PluginMetadata {
                let mut meta = PluginMetadata::new(&self.id, "1.0.0");
                for dep in &self.deps {
                    meta = meta.with_dependency(dep.clone(), ">=0");
                }
                meta
            }

            fn on_unload(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
                let _ = ctx;
                Ok(())
            }

            fn as_any(&self) -> &dyn Any {
                self
            }

            fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }
        }

        let registry = PluginRegistry::new();
        registry
            .register(Box::new(DepPlugin {
                id: "a".to_string(),
                deps: Vec::new(),
            }))
            .unwrap();
        registry
            .register(Box::new(DepPlugin {
                id: "b".to_string(),
                deps: vec!["a".to_string()],
            }))
            .unwrap();
        registry
            .register(Box::new(DepPlugin {
                id: "c".to_string(),
                deps: vec!["b".to_string()],
            }))
            .unwrap();

        let unloaded = registry.unregister_safe("a", true).unwrap();
        assert_eq!(
            unloaded,
            vec!["c".to_string(), "b".to_string(), "a".to_string()]
        );
        assert!(registry.list_plugins().is_empty());

        let unloaded_missing = registry.unregister_safe("missing", false).unwrap();
        assert!(unloaded_missing.is_empty());
        assert!(registry.unregister_safe("missing", true).is_err());
    }
}
