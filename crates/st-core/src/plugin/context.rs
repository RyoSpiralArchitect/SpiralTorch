// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Plugin context providing access to the runtime environment.

use super::events::{PluginEventBus, EventListener};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Dependency specification for a plugin.
#[derive(Debug, Clone)]
pub struct PluginDependency {
    /// Plugin ID
    pub plugin_id: String,
    /// Version requirement (semver-compatible string)
    pub version_req: String,
}

/// Context provided to plugins during lifecycle events.
///
/// This gives plugins access to the event bus, configuration, and other plugins.
pub struct PluginContext {
    /// Event bus for pub/sub messaging
    pub event_bus: PluginEventBus,
    /// Shared configuration key-value store
    config: Arc<Mutex<HashMap<String, String>>>,
    /// Registry of services provided by plugins
    services: Arc<Mutex<HashMap<String, Arc<dyn std::any::Any + Send + Sync>>>>,
}

impl PluginContext {
    /// Create a new plugin context.
    pub fn new(event_bus: PluginEventBus) -> Self {
        Self {
            event_bus,
            config: Arc::new(Mutex::new(HashMap::new())),
            services: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get a configuration value.
    pub fn get_config(&self, key: &str) -> Option<String> {
        self.config.lock().unwrap().get(key).cloned()
    }

    /// Set a configuration value.
    pub fn set_config(&self, key: impl Into<String>, value: impl Into<String>) {
        self.config.lock().unwrap().insert(key.into(), value.into());
    }

    /// Subscribe to an event type.
    pub fn subscribe(&self, event_type: impl Into<String>, listener: EventListener) {
        self.event_bus.subscribe(event_type, listener);
    }

    /// Register a service that other plugins can access.
    pub fn register_service<T: std::any::Any + Send + Sync + 'static>(
        &self,
        name: impl Into<String>,
        service: T,
    ) {
        self.services.lock().unwrap().insert(name.into(), Arc::new(service));
    }

    /// Get a service registered by another plugin.
    pub fn get_service<T: std::any::Any + Send + Sync + 'static>(
        &self,
        name: &str,
    ) -> Option<Arc<T>> {
        self.services
            .lock()
            .unwrap()
            .get(name)
            .and_then(|s| s.clone().downcast::<T>().ok())
    }

    /// List all registered services.
    pub fn list_services(&self) -> Vec<String> {
        self.services.lock().unwrap().keys().cloned().collect()
    }
}

impl Clone for PluginContext {
    fn clone(&self) -> Self {
        Self {
            event_bus: self.event_bus.clone(),
            config: Arc::clone(&self.config),
            services: Arc::clone(&self.services),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_storage() {
        let ctx = PluginContext::new(PluginEventBus::new());
        
        ctx.set_config("key1", "value1");
        assert_eq!(ctx.get_config("key1"), Some("value1".to_string()));
        assert_eq!(ctx.get_config("key2"), None);
    }

    #[test]
    fn test_service_registration() {
        let ctx = PluginContext::new(PluginEventBus::new());
        
        ctx.register_service("test_service", 42i32);
        
        let service = ctx.get_service::<i32>("test_service");
        assert!(service.is_some());
        assert_eq!(*service.unwrap(), 42);
    }
}
