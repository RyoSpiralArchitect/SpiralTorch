// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

#![cfg_attr(
    not(test),
    deny(clippy::expect_used, clippy::panic, clippy::unwrap_used)
)]

//! Dynamic module discovery and composition system.
//!
//! This module provides mechanisms for discovering, loading, and composing
//! neural network modules at runtime, enabling flexible model architectures.

use crate::{Module, PureResult};
use st_core::plugin::{PluginEvent, PluginRegistry};
use st_core::TensorError;
use std::any::Any;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard};

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            mutex.clear_poison();
            poisoned.into_inner()
        }
    }
}

fn panic_payload_message(payload: Box<dyn Any + Send>) -> String {
    let payload = match payload.downcast::<String>() {
        Ok(message) => return *message,
        Err(payload) => payload,
    };
    let payload = match payload.downcast::<&'static str>() {
        Ok(message) => return (*message).to_string(),
        Err(payload) => payload,
    };

    if let Err(secondary_payload) = catch_unwind(AssertUnwindSafe(|| drop(payload))) {
        std::mem::forget(secondary_payload);
    }
    "non-string panic payload".to_string()
}

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
pub type ModuleFactory =
    Arc<dyn Fn(&HashMap<String, String>) -> PureResult<Box<dyn Module>> + Send + Sync>;

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
    pub fn register(&self, metadata: ModuleMetadata, factory: ModuleFactory) -> PureResult<()> {
        let module_type = metadata.module_type.clone();
        if module_type.trim().is_empty() {
            return Err(TensorError::Generic(
                "Module type must not be empty".to_string(),
            ));
        }

        {
            let mut factories = lock_recover(&self.factories);
            if factories.contains_key(&module_type) {
                return Err(TensorError::Generic(format!(
                    "Module type '{}' already registered",
                    module_type
                )));
            }

            factories.insert(module_type.clone(), (metadata, factory));
        }

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
        let factory = {
            let factories = lock_recover(&self.factories);
            let (_, factory) = factories.get(module_type).ok_or_else(|| {
                TensorError::Generic(format!("Module type '{}' not found", module_type))
            })?;
            Arc::clone(factory)
        };

        match catch_unwind(AssertUnwindSafe(|| factory(config))) {
            Ok(result) => result,
            Err(payload) => Err(TensorError::Generic(format!(
                "Module factory '{}' panicked: {}",
                module_type,
                panic_payload_message(payload)
            ))),
        }
    }

    /// Get metadata for a module type.
    pub fn get_metadata(&self, module_type: &str) -> Option<ModuleMetadata> {
        lock_recover(&self.factories)
            .get(module_type)
            .map(|(meta, _)| meta.clone())
    }

    /// List all registered module types.
    pub fn list_modules(&self) -> Vec<String> {
        let mut modules = lock_recover(&self.factories)
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        modules.sort();
        modules
    }

    /// Find modules by category.
    pub fn find_by_category(&self, category: &ModuleCategory) -> Vec<ModuleMetadata> {
        let mut modules = lock_recover(&self.factories)
            .values()
            .filter(|(meta, _)| &meta.category == category)
            .map(|(meta, _)| meta.clone())
            .collect::<Vec<_>>();
        modules.sort_by(|left, right| left.module_type.cmp(&right.module_type));
        modules
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
    ($registry:expr, $module_type:expr, $category:expr, $factory:expr) => {{
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
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    fn metadata(module_type: &str, category: ModuleCategory) -> ModuleMetadata {
        ModuleMetadata {
            module_type: module_type.to_string(),
            display_name: module_type.to_string(),
            description: String::new(),
            input_shape: None,
            output_shape: None,
            parameters: HashMap::new(),
            category,
        }
    }

    fn error_factory(message: &'static str) -> ModuleFactory {
        Arc::new(move |_: &HashMap<String, String>| Err(TensorError::Generic(message.to_string())))
    }

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

    #[test]
    fn plugin_listener_can_query_registry_during_registration() {
        let plugin_registry = Arc::new(PluginRegistry::new());
        let registry = Arc::new(ModuleDiscoveryRegistry::with_plugin_registry(Arc::clone(
            &plugin_registry,
        )));
        let listener_registry = Arc::clone(&registry);
        let (sender, receiver) = mpsc::channel();
        plugin_registry.event_bus().subscribe(
            "ModuleRegistered",
            Arc::new(move |_| {
                let _ = sender.send(listener_registry.list_modules());
            }),
        );

        let worker_registry = Arc::clone(&registry);
        let worker = std::thread::spawn(move || {
            worker_registry.register(
                metadata("listener_visible", ModuleCategory::Layer),
                error_factory("unused"),
            )
        });

        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
            vec!["listener_visible".to_string()]
        );
        worker.join().unwrap().unwrap();
    }

    #[test]
    fn factory_can_query_registry_without_deadlocking() {
        let registry = Arc::new(ModuleDiscoveryRegistry::new());
        let weak_registry = Arc::downgrade(&registry);
        let (sender, receiver) = mpsc::channel();
        let factory: ModuleFactory = Arc::new(move |_| {
            let modules = weak_registry
                .upgrade()
                .map(|registry| registry.list_modules())
                .unwrap_or_default();
            let _ = sender.send(modules);
            Err(TensorError::Generic("factory finished".to_string()))
        });
        registry
            .register(
                metadata("reentrant", ModuleCategory::Custom("test".into())),
                factory,
            )
            .unwrap();

        let worker_registry = Arc::clone(&registry);
        let worker = std::thread::spawn(move || {
            worker_registry
                .create("reentrant", &HashMap::new())
                .err()
                .map(|error| error.to_string())
        });

        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(1)).unwrap(),
            vec!["reentrant".to_string()]
        );
        assert_eq!(worker.join().unwrap().as_deref(), Some("factory finished"));
    }

    #[test]
    fn factory_panic_is_returned_without_poisoning_registry() {
        let registry = ModuleDiscoveryRegistry::new();
        let factory: ModuleFactory = Arc::new(|_| panic!("factory boom"));
        registry
            .register(metadata("panics", ModuleCategory::Layer), factory)
            .unwrap();

        let error = registry.create("panics", &HashMap::new()).err().unwrap();

        assert!(error
            .to_string()
            .contains("factory 'panics' panicked: factory boom"));
        assert_eq!(registry.list_modules(), vec!["panics".to_string()]);
    }

    #[test]
    fn panicking_factory_payload_destructor_is_contained() {
        struct PanicOnDrop;
        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                panic!("payload drop panic");
            }
        }

        let registry = ModuleDiscoveryRegistry::new();
        let factory: ModuleFactory = Arc::new(|_| std::panic::panic_any(PanicOnDrop));
        registry
            .register(metadata("payload", ModuleCategory::Layer), factory)
            .unwrap();

        let error = registry.create("payload", &HashMap::new()).err().unwrap();

        assert!(error.to_string().contains("non-string panic payload"));
        assert_eq!(registry.list_modules(), vec!["payload".to_string()]);
    }

    #[test]
    fn poisoned_registry_lock_is_recovered() {
        let registry = ModuleDiscoveryRegistry::new();
        let _ = catch_unwind(AssertUnwindSafe(|| {
            let _guard = registry.factories.lock().unwrap();
            panic!("poison registry");
        }));
        assert!(registry.factories.is_poisoned());

        registry
            .register(
                metadata("recovered", ModuleCategory::Layer),
                error_factory("unused"),
            )
            .unwrap();

        assert!(!registry.factories.is_poisoned());
        assert_eq!(registry.list_modules(), vec!["recovered".to_string()]);
    }

    #[test]
    fn module_names_and_category_results_are_deterministic() {
        let registry = ModuleDiscoveryRegistry::new();
        for module_type in ["zeta", "alpha", "middle"] {
            registry
                .register(
                    metadata(module_type, ModuleCategory::Layer),
                    error_factory("unused"),
                )
                .unwrap();
        }

        assert_eq!(
            registry.list_modules(),
            vec![
                "alpha".to_string(),
                "middle".to_string(),
                "zeta".to_string()
            ]
        );
        assert_eq!(
            registry
                .find_by_category(&ModuleCategory::Layer)
                .into_iter()
                .map(|metadata| metadata.module_type)
                .collect::<Vec<_>>(),
            vec![
                "alpha".to_string(),
                "middle".to_string(),
                "zeta".to_string()
            ]
        );
    }

    #[test]
    fn empty_module_type_is_rejected() {
        let registry = ModuleDiscoveryRegistry::new();
        let error = registry
            .register(
                metadata("  ", ModuleCategory::Layer),
                error_factory("unused"),
            )
            .unwrap_err();

        assert!(error.to_string().contains("must not be empty"));
        assert!(registry.list_modules().is_empty());
    }
}
