// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Bridge between the unified plugin system and ZSpaceSequencerPlugin.
//!
//! This module allows ZSpaceSequencerPlugins to be registered in the global
//! plugin registry, enabling organic ecosystem integration.

use st_core::plugin::{Plugin, PluginCapability, PluginContext, PluginMetadata};
use st_core::PureResult;
use crate::zspace_coherence::ZSpaceSequencerPlugin;
use std::any::Any;
use std::sync::{Arc, Mutex};

/// Wrapper that adapts a ZSpaceSequencerPlugin to the unified Plugin interface.
pub struct ZSpacePluginAdapter {
    inner: Arc<Mutex<Box<dyn ZSpaceSequencerPlugin>>>,
    metadata: PluginMetadata,
}

impl ZSpacePluginAdapter {
    /// Create a new adapter for a ZSpaceSequencerPlugin.
    pub fn new(
        plugin: Box<dyn ZSpaceSequencerPlugin>,
        metadata: PluginMetadata,
    ) -> Self {
        Self {
            inner: Arc::new(Mutex::new(plugin)),
            metadata,
        }
    }

    /// Create with auto-generated metadata based on the plugin name.
    pub fn with_auto_metadata(plugin: Box<dyn ZSpaceSequencerPlugin>) -> Self {
        let name = plugin.name();
        let metadata = PluginMetadata::new(
            format!("zspace_sequencer_{}", name),
            "1.0.0"
        )
            .with_name(format!("ZSpace Sequencer: {}", name))
            .with_description(format!("ZSpace coherence sequencing plugin: {}", name))
            .with_capability(PluginCapability::Custom("ZSpaceSequencer".to_string()));

        Self::new(plugin, metadata)
    }

    /// Get a reference to the inner ZSpaceSequencerPlugin.
    pub fn inner(&self) -> Arc<Mutex<Box<dyn ZSpaceSequencerPlugin>>> {
        Arc::clone(&self.inner)
    }
}

impl Plugin for ZSpacePluginAdapter {
    fn metadata(&self) -> PluginMetadata {
        self.metadata.clone()
    }

    fn on_load(&mut self, ctx: &mut PluginContext) -> PureResult<()> {
        // Register the sequencer plugin as a service so others can access it
        let service_name = format!("zspace_sequencer_{}", self.metadata.id);
        ctx.register_service(&service_name, Arc::clone(&self.inner));
        
        println!("✅ Loaded ZSpace sequencer plugin: {}", self.metadata.id);
        Ok(())
    }

    fn on_unload(&mut self, _ctx: &mut PluginContext) -> PureResult<()> {
        println!("👋 Unloading ZSpace sequencer plugin: {}", self.metadata.id);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Helper to register ZSpaceSequencerPlugins with the global plugin registry.
pub struct ZSpacePluginRegistry;

impl ZSpacePluginRegistry {
    /// Register a ZSpaceSequencerPlugin with custom metadata.
    pub fn register_with_metadata(
        plugin: Box<dyn ZSpaceSequencerPlugin>,
        metadata: PluginMetadata,
    ) -> PureResult<()> {
        let adapter = Box::new(ZSpacePluginAdapter::new(plugin, metadata));
        st_core::plugin::global_registry().register(adapter)
    }

    /// Register a ZSpaceSequencerPlugin with auto-generated metadata.
    pub fn register(plugin: Box<dyn ZSpaceSequencerPlugin>) -> PureResult<()> {
        let adapter = Box::new(ZSpacePluginAdapter::with_auto_metadata(plugin));
        st_core::plugin::global_registry().register(adapter)
    }

    /// Find all registered ZSpace sequencer plugins.
    pub fn find_all_sequencers() -> Vec<Arc<Mutex<Box<dyn ZSpaceSequencerPlugin>>>> {
        let registry = st_core::plugin::global_registry();
        let handles = registry.find_by_capability(
            &PluginCapability::Custom("ZSpaceSequencer".to_string())
        );

        handles
            .iter()
            .filter_map(|handle| {
                handle.with_plugin(|plugin| {
                    plugin
                        .as_any()
                        .downcast_ref::<ZSpacePluginAdapter>()
                        .map(|adapter| adapter.inner())
                })
            })
            .collect()
    }

    /// Get a specific ZSpace sequencer plugin by name.
    pub fn get_sequencer(name: &str) -> Option<Arc<Mutex<Box<dyn ZSpaceSequencerPlugin>>>> {
        let plugin_id = format!("zspace_sequencer_{}", name);
        let registry = st_core::plugin::global_registry();
        
        registry.get(&plugin_id).and_then(|handle| {
            handle.with_plugin(|plugin| {
                plugin
                    .as_any()
                    .downcast_ref::<ZSpacePluginAdapter>()
                    .map(|adapter| adapter.inner())
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zspace_coherence::{ZSpaceSequencerPlugin, ZSpaceSequencerStage};

    struct TestSequencerPlugin;

    impl ZSpaceSequencerPlugin for TestSequencerPlugin {
        fn name(&self) -> &'static str {
            "test_sequencer"
        }

        fn on_stage(&self, _stage: ZSpaceSequencerStage<'_>) -> PureResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_adapter_creation() {
        let plugin = Box::new(TestSequencerPlugin);
        let adapter = ZSpacePluginAdapter::with_auto_metadata(plugin);
        
        let meta = adapter.metadata();
        assert!(meta.id.contains("test_sequencer"));
    }
}
