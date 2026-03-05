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

    /// Unset a configuration value.
    ///
    /// Returns `true` when a key existed and was removed.
    pub fn unset_config(&self, key: &str) -> bool {
        self.config.lock().unwrap().remove(key).is_some()
    }

    /// List all configuration key/value pairs.
    ///
    /// The returned list is sorted by key for deterministic iteration.
    pub fn list_config(&self) -> Vec<(String, String)> {
        let mut items: Vec<(String, String)> = self
            .config
            .lock()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        items
    }

    /// Clear configuration values, optionally filtering by key prefix.
    ///
    /// Returns a sorted list of keys that were removed.
    pub fn clear_config(&self, prefix: Option<&str>) -> Vec<String> {
        let mut config = self.config.lock().unwrap();
        let mut keys: Vec<String> = config.keys().cloned().collect();
        keys.sort();

        let mut removed = Vec::new();
        for key in keys {
            if let Some(prefix) = prefix {
                if !key.starts_with(prefix) {
                    continue;
                }
            }
            if config.remove(&key).is_some() {
                removed.push(key);
            }
        }
        removed
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

    /// Unregister a previously registered service.
    pub fn unregister_service(&self, name: &str) -> bool {
        self.services.lock().unwrap().remove(name).is_some()
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
        ctx.set_config("prefix.a", "1");
        ctx.set_config("prefix.b", "2");
        assert_eq!(ctx.get_config("key1"), Some("value1".to_string()));
        assert_eq!(ctx.get_config("key2"), None);

        assert_eq!(
            ctx.list_config(),
            vec![
                ("key1".to_string(), "value1".to_string()),
                ("prefix.a".to_string(), "1".to_string()),
                ("prefix.b".to_string(), "2".to_string())
            ]
        );

        assert!(ctx.unset_config("key1"));
        assert!(!ctx.unset_config("key1"));
        assert_eq!(ctx.get_config("key1"), None);

        assert_eq!(
            ctx.clear_config(Some("prefix.")),
            vec!["prefix.a".to_string(), "prefix.b".to_string()]
        );
        assert!(ctx.list_config().is_empty());
    }

    #[test]
    fn test_service_registration() {
        let ctx = PluginContext::new(PluginEventBus::new());
        
        ctx.register_service("test_service", 42i32);
        
        let service = ctx.get_service::<i32>("test_service");
        assert!(service.is_some());
        assert_eq!(*service.unwrap(), 42);
        assert!(ctx.unregister_service("test_service"));
        assert!(ctx.get_service::<i32>("test_service").is_none());
        assert!(!ctx.unregister_service("test_service"));
    }
}
