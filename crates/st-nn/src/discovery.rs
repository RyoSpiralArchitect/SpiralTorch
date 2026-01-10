// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Dynamic module discovery and composition system.
//!
//! This module provides mechanisms for discovering, loading, and composing
//! neural network modules at runtime, enabling flexible model architectures.

use crate::{Module, PureResult};
use st_core::plugin::{PluginEvent, PluginRegistry};
use st_core::TensorError;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Metadata describing a discoverable module.
#[derive(Debug, Clone)]
pub struct ModuleMetadata {
    /// Unique identifier for this module type
    pub module_type: String,
    /// Human-readable name
    pub display_name: String,
    /// Brief description
    pub description: String,
    /// Input shape requirements (None means flexible)
    pub input_shape: Option<Vec<usize>>,
    /// Output shape (None means depends on input)
    pub output_shape: Option<Vec<usize>>,
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    /// Category (layer, loss, optimizer, etc.)
    pub category: ModuleCategory,
}

/// Category of module.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ModuleCategory {
    Layer,
    Loss,
    Optimizer,
    Activation,
    Normalization,
    Attention,
    Convolution,
    Pooling,
    Embedding,
    Encoder,
    Decoder,
    Custom(String),
}

/// Factory function that creates a module instance.
pub type ModuleFactory = Arc<dyn Fn(&HashMap<String, String>) -> PureResult<Box<dyn Module>> + Send + Sync>;

/// Registry for discoverable modules.
pub struct ModuleDiscoveryRegistry {
    factories: Arc<Mutex<HashMap<String, (ModuleMetadata, ModuleFactory)>>>,
    plugin_registry: Option<Arc<PluginRegistry>>,
}

impl ModuleDiscoveryRegistry {
    /// Create a new module discovery registry.
    pub fn new() -> Self {
        Self {
            factories: Arc::new(Mutex::new(HashMap::new())),
            plugin_registry: None,
        }
    }

    /// Create with a plugin registry for ecosystem integration.
    pub fn with_plugin_registry(plugin_registry: Arc<PluginRegistry>) -> Self {
        Self {
            factories: Arc::new(Mutex::new(HashMap::new())),
            plugin_registry: Some(plugin_registry),
        }
    }

    /// Register a module factory.
    pub fn register(
        &self,
        metadata: ModuleMetadata,
        factory: ModuleFactory,
    ) -> PureResult<()> {
        let module_type = metadata.module_type.clone();
        
        let mut factories = self.factories.lock().unwrap();
        if factories.contains_key(&module_type) {
            return Err(TensorError::Generic(format!(
                "Module type '{}' already registered",
                module_type
            )));
        }
        
        factories.insert(module_type.clone(), (metadata, factory));
        
        // Notify plugin system if available
        if let Some(registry) = &self.plugin_registry {
            registry.event_bus().publish(&PluginEvent::Custom {
                event_type: "ModuleRegistered".to_string(),
                data: Arc::new(module_type),
            });
        }
        
        Ok(())
    }

    /// Create a module by type with configuration.
    pub fn create(
        &self,
        module_type: &str,
        config: &HashMap<String, String>,
    ) -> PureResult<Box<dyn Module>> {
        let factories = self.factories.lock().unwrap();
        
        let (_, factory) = factories
            .get(module_type)
            .ok_or_else(|| TensorError::Generic(format!(
                "Module type '{}' not found",
                module_type
            )))?;
        
        factory(config)
    }

    /// Get metadata for a module type.
    pub fn get_metadata(&self, module_type: &str) -> Option<ModuleMetadata> {
        self.factories
            .lock()
            .unwrap()
            .get(module_type)
            .map(|(meta, _)| meta.clone())
    }

    /// List all registered module types.
    pub fn list_modules(&self) -> Vec<String> {
        self.factories.lock().unwrap().keys().cloned().collect()
    }

    /// Find modules by category.
    pub fn find_by_category(&self, category: &ModuleCategory) -> Vec<ModuleMetadata> {
        self.factories
            .lock()
            .unwrap()
            .values()
            .filter(|(meta, _)| &meta.category == category)
            .map(|(meta, _)| meta.clone())
            .collect()
    }
}

impl Default for ModuleDiscoveryRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating module pipelines dynamically.
pub struct ModulePipelineBuilder {
    registry: Arc<ModuleDiscoveryRegistry>,
    modules: Vec<Box<dyn Module>>,
    name: Option<String>,
}

impl ModulePipelineBuilder {
    /// Create a new pipeline builder.
    pub fn new(registry: Arc<ModuleDiscoveryRegistry>) -> Self {
        Self {
            registry,
            modules: Vec::new(),
            name: None,
        }
    }

    /// Set the pipeline name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Add a module by type and configuration.
    pub fn add_module(
        mut self,
        module_type: &str,
        config: HashMap<String, String>,
    ) -> PureResult<Self> {
        let module = self.registry.create(module_type, &config)?;
        self.modules.push(module);
        Ok(self)
    }

    /// Add a pre-constructed module.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, module: Box<dyn Module>) -> Self {
        self.modules.push(module);
        self
    }

    /// Build a sequential pipeline from the added modules.
    pub fn build(self) -> PureResult<crate::Sequential> {
        let mut sequential = crate::Sequential::new();
        for module in self.modules {
            sequential.push_boxed(module);
        }
        Ok(sequential)
    }
}

/// Helper macros for registering modules.
#[macro_export]
macro_rules! register_module {
    ($registry:expr, $module_type:expr, $category:expr, $factory:expr) => {
        {
            let metadata = $crate::discovery::ModuleMetadata {
                module_type: $module_type.to_string(),
                display_name: $module_type.to_string(),
                description: String::new(),
                input_shape: None,
                output_shape: None,
                parameters: std::collections::HashMap::new(),
                category: $category,
            };
            
            $registry.register(metadata, std::sync::Arc::new($factory))
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_registration() {
        let registry = ModuleDiscoveryRegistry::new();
        
        let metadata = ModuleMetadata {
            module_type: "test_module".to_string(),
            display_name: "Test Module".to_string(),
            description: "A test module".to_string(),
            input_shape: Some(vec![10]),
            output_shape: Some(vec![5]),
            parameters: HashMap::new(),
            category: ModuleCategory::Layer,
        };
        
        let factory = Arc::new(|_config: &HashMap<String, String>| {
            // Would create a real module here
            Err(TensorError::Generic("Not implemented in test".to_string()))
        });
        
        assert!(registry.register(metadata, factory).is_ok());
        assert!(registry.get_metadata("test_module").is_some());
    }

    #[test]
    fn test_find_by_category() {
        let registry = ModuleDiscoveryRegistry::new();
        
        let metadata1 = ModuleMetadata {
            module_type: "layer1".to_string(),
            display_name: "Layer 1".to_string(),
            description: "".to_string(),
            input_shape: None,
            output_shape: None,
            parameters: HashMap::new(),
            category: ModuleCategory::Layer,
        };
        
        let metadata2 = ModuleMetadata {
            module_type: "activation1".to_string(),
            display_name: "Activation 1".to_string(),
            description: "".to_string(),
            input_shape: None,
            output_shape: None,
            parameters: HashMap::new(),
            category: ModuleCategory::Activation,
        };
        
        let factory = Arc::new(|_: &HashMap<String, String>| {
            Err(TensorError::Generic("Not implemented".to_string()))
        });
        
        registry.register(metadata1, factory.clone()).unwrap();
        registry.register(metadata2, factory).unwrap();
        
        let layers = registry.find_by_category(&ModuleCategory::Layer);
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].module_type, "layer1");
    }
}
