// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Plugin registry for managing loaded plugins.

use super::context::PluginContext;
use super::events::{PluginEvent, PluginEventBus};
use super::traits::{Plugin, PluginCapability, PluginMetadata};
use crate::{PureResult, TensorError};
use std::collections::HashMap;
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

        // Call on_load hook
        let mut ctx = self.context.lock().unwrap();
        plugin.on_load(&mut ctx)?;
        drop(ctx);

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
        let mut ctx = self.context.lock().unwrap();
        handle.with_plugin(|plugin: &mut dyn Plugin| plugin.on_unload(&mut ctx))?;
        drop(ctx);

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

        let plugin_ids: Vec<_> = self.list_plugins();
        for plugin_id in plugin_ids {
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
        for (dep_id, _version_req) in &metadata.dependencies {
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
}
