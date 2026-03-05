// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Plugin registry for managing loaded plugins.

use super::context::PluginContext;
use super::events::{PluginEvent, PluginEventBus};
use super::traits::{Plugin, PluginCapability, PluginMetadata};
use crate::{PureResult, TensorError};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};

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
        self.plugins.write().unwrap().insert(plugin_id.clone(), handle);

        // Emit event
        self.event_bus.publish(&PluginEvent::PluginLoaded { plugin_id });

        Ok(())
    }

    /// Unregister a plugin by ID.
    pub fn unregister(&self, plugin_id: &str) -> PureResult<()> {
        let handle = {
            let mut plugins = self.plugins.write().unwrap();
            plugins.remove(plugin_id).ok_or_else(|| {
                TensorError::Generic(format!("Plugin '{}' not found", plugin_id))
            })?
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
        let mut edges: HashMap<String, Vec<String>> =
            plugin_ids.iter().map(|id| (id.clone(), Vec::new())).collect();

        for id in &plugin_ids {
            let deps = deps_by_id.get(id).map(|deps| deps.as_slice()).unwrap_or(&[]);
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
            PluginMetadata::new(&self.name, "1.0.0")
                .with_capability(PluginCapability::Operators)
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
        assert_eq!(unloaded, vec!["b".to_string(), "c".to_string(), "a".to_string()]);
        assert!(registry.list_plugins().is_empty());
    }
}
