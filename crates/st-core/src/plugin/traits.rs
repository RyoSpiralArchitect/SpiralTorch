// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Core traits and types for the plugin system.

use crate::PureResult;
use super::context::PluginContext;
use super::events::PluginEvent;
use std::any::Any;
use std::collections::HashMap;

/// Metadata describing a plugin.
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    /// Unique identifier for the plugin
    pub id: String,
    /// Semantic version string
    pub version: String,
    /// Human-readable name
    pub name: Option<String>,
    /// Brief description of functionality
    pub description: Option<String>,
    /// Plugin author/maintainer
    pub author: Option<String>,
    /// Dependencies on other plugins (id -> version requirement)
    pub dependencies: HashMap<String, String>,
    /// Capabilities provided by this plugin
    pub capabilities: Vec<PluginCapability>,
    /// Arbitrary metadata
    pub metadata: HashMap<String, String>,
}

impl PluginMetadata {
    /// Create new plugin metadata with required fields.
    pub fn new(id: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            version: version.into(),
            name: None,
            description: None,
            author: None,
            dependencies: HashMap::new(),
            capabilities: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Set the plugin name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the plugin description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the plugin author.
    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    /// Add a dependency on another plugin.
    pub fn with_dependency(mut self, plugin_id: impl Into<String>, version: impl Into<String>) -> Self {
        self.dependencies.insert(plugin_id.into(), version.into());
        self
    }

    /// Add a capability provided by this plugin.
    pub fn with_capability(mut self, cap: PluginCapability) -> Self {
        self.capabilities.push(cap);
        self
    }

    /// Add custom metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Capabilities that a plugin can provide.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PluginCapability {
    /// Provides a backend implementation
    Backend(String),
    /// Provides operators/kernels
    Operators,
    /// Provides loss functions
    LossFunctions,
    /// Provides optimizers
    Optimizers,
    /// Provides data loaders
    DataLoaders,
    /// Provides visualizations
    Visualization,
    /// Provides telemetry/monitoring
    Telemetry,
    /// Provides language models/NLP
    Language,
    /// Provides computer vision
    Vision,
    /// Provides RL environments/agents
    ReinforcementLearning,
    /// Provides graph neural network components
    GraphNeuralNetworks,
    /// Provides recommendation system components
    Recommender,
    /// Custom capability
    Custom(String),
}

/// Core trait that all plugins must implement.
///
/// Plugins are the primary extension mechanism in SpiralTorch. They can provide
/// new operators, backends, loss functions, optimizers, and more.
pub trait Plugin: Send + Sync {
    /// Returns metadata describing this plugin.
    fn metadata(&self) -> PluginMetadata;

    /// Called when the plugin is loaded into the registry.
    ///
    /// This is where the plugin should register itself with the context,
    /// declare its capabilities, and perform any initialization.
    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        let _ = ctx;
        Ok(())
    }

    /// Called before the plugin is unloaded from the registry.
    ///
    /// This is where the plugin should clean up resources and deregister
    /// any callbacks or handlers.
    fn on_unload(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        let _ = ctx;
        Ok(())
    }

    /// Handle an event from the event bus.
    ///
    /// Plugins can subscribe to events during `on_load` and will receive
    /// notifications via this method.
    fn on_event(&mut self, event: &PluginEvent, ctx: &PluginContext) -> PureResult<()> {
        let _ = (event, ctx);
        Ok(())
    }

    /// Cast this plugin to Any for downcasting.
    ///
    /// This allows plugins to expose custom interfaces beyond the base trait.
    fn as_any(&self) -> &dyn Any;

    /// Cast this plugin to mutable Any for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Extension trait for querying plugin capabilities.
pub trait PluginCapabilityExt {
    /// Check if the plugin provides a specific capability.
    fn has_capability(&self, cap: &PluginCapability) -> bool;

    /// Get all capabilities provided by this plugin.
    fn capabilities(&self) -> Vec<PluginCapability>;
}

impl<P: Plugin + ?Sized> PluginCapabilityExt for P {
    fn has_capability(&self, cap: &PluginCapability) -> bool {
        self.metadata().capabilities.contains(cap)
    }

    fn capabilities(&self) -> Vec<PluginCapability> {
        self.metadata().capabilities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin;

    impl Plugin for TestPlugin {
        fn metadata(&self) -> PluginMetadata {
            PluginMetadata::new("test_plugin", "1.0.0")
                .with_name("Test Plugin")
                .with_description("A test plugin")
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
    fn test_plugin_metadata() {
        let plugin = TestPlugin;
        let meta = plugin.metadata();
        
        assert_eq!(meta.id, "test_plugin");
        assert_eq!(meta.version, "1.0.0");
        assert_eq!(meta.name, Some("Test Plugin".to_string()));
        assert!(plugin.has_capability(&PluginCapability::Operators));
        assert!(!plugin.has_capability(&PluginCapability::Vision));
    }
}
