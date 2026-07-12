// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Plugin context providing access to the runtime environment.

use super::events::{EventListener, PluginEventBus};
use super::sync::lock_recover;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

type ErasedPluginService = Arc<dyn std::any::Any + Send + Sync>;
type PluginServiceMap = HashMap<String, ErasedPluginService>;

/// Dependency specification for a plugin.
#[derive(Debug, Clone)]
pub struct PluginDependency {
    /// Plugin ID
    pub plugin_id: String,
    /// Version requirement (semver-compatible string)
    pub version_req: String,
}

/// Ownership token for a service registered in a [`PluginContext`].
///
/// Dropping the token unregisters the service only when the same registration is still current.
/// A newer service installed under the same name is left intact.
#[must_use = "dropping the token unregisters the owned plugin service"]
pub struct PluginServiceRegistration {
    name: String,
    service: Weak<dyn std::any::Any + Send + Sync>,
    services: Weak<Mutex<PluginServiceMap>>,
}

impl PluginServiceRegistration {
    /// Returns the registered service name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for PluginServiceRegistration {
    fn drop(&mut self) {
        let Some(services) = self.services.upgrade() else {
            return;
        };
        let Some(service) = self.service.upgrade() else {
            return;
        };
        let removed = {
            let mut services = lock_recover(&services);
            let owns_current = services
                .get(&self.name)
                .is_some_and(|current| Arc::ptr_eq(current, &service));
            if owns_current {
                services.remove(&self.name)
            } else {
                None
            }
        };
        drop(removed);
    }
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
    services: Arc<Mutex<PluginServiceMap>>,
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
        lock_recover(&self.config).get(key).cloned()
    }

    /// Set a configuration value.
    pub fn set_config(&self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        lock_recover(&self.config).insert(key, value);
    }

    /// Unset a configuration value.
    ///
    /// Returns `true` when a key existed and was removed.
    pub fn unset_config(&self, key: &str) -> bool {
        lock_recover(&self.config).remove(key).is_some()
    }

    /// List all configuration key/value pairs.
    ///
    /// The returned list is sorted by key for deterministic iteration.
    pub fn list_config(&self) -> Vec<(String, String)> {
        let mut items: Vec<(String, String)> = lock_recover(&self.config)
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
        let mut config = lock_recover(&self.config);
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
        let name = name.into();
        let service: ErasedPluginService = Arc::new(service);
        let previous = {
            let mut services = lock_recover(&self.services);
            services.insert(name, service)
        };
        drop(previous);
    }

    /// Registers a service whose lifetime is owned by the returned token.
    ///
    /// Dropping the token removes this service when it is still current. If another component
    /// replaces the same name first, dropping the older token preserves the replacement.
    pub fn register_owned_service<T: std::any::Any + Send + Sync + 'static>(
        &self,
        name: impl Into<String>,
        service: T,
    ) -> PluginServiceRegistration {
        let name = name.into();
        let service: ErasedPluginService = Arc::new(service);
        let registration = PluginServiceRegistration {
            name: name.clone(),
            service: Arc::downgrade(&service),
            services: Arc::downgrade(&self.services),
        };
        let previous = {
            let mut services = lock_recover(&self.services);
            services.insert(name, service)
        };
        drop(previous);
        registration
    }

    /// Get a service registered by another plugin.
    pub fn get_service<T: std::any::Any + Send + Sync + 'static>(
        &self,
        name: &str,
    ) -> Option<Arc<T>> {
        lock_recover(&self.services)
            .get(name)
            .and_then(|s| s.clone().downcast::<T>().ok())
    }

    /// List all registered services.
    pub fn list_services(&self) -> Vec<String> {
        let mut names: Vec<String> = lock_recover(&self.services).keys().cloned().collect();
        names.sort();
        names
    }

    /// Unregister a previously registered service.
    pub fn unregister_service(&self, name: &str) -> bool {
        let removed = {
            let mut services = lock_recover(&self.services);
            services.remove(name)
        };
        removed.is_some()
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
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Weak;
    use std::thread;

    struct PanickingString;

    impl From<PanickingString> for String {
        fn from(_: PanickingString) -> Self {
            panic!("conversion failed")
        }
    }

    struct ServiceDropProbe {
        services: Weak<Mutex<HashMap<String, Arc<dyn std::any::Any + Send + Sync>>>>,
        dropped_without_lock: Arc<AtomicBool>,
    }

    impl Drop for ServiceDropProbe {
        fn drop(&mut self) {
            let unlocked = self
                .services
                .upgrade()
                .is_some_and(|services| services.try_lock().is_ok());
            self.dropped_without_lock.store(unlocked, Ordering::SeqCst);
        }
    }

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
        ctx.register_service("alpha_service", 7i32);

        let service = ctx.get_service::<i32>("test_service");
        assert!(service.is_some());
        assert_eq!(*service.unwrap(), 42);
        assert_eq!(
            ctx.list_services(),
            vec!["alpha_service".to_string(), "test_service".to_string()]
        );
        assert!(ctx.unregister_service("test_service"));
        assert!(ctx.get_service::<i32>("test_service").is_none());
        assert!(!ctx.unregister_service("test_service"));
    }

    #[test]
    fn test_owned_service_registration_tracks_identity() {
        let ctx = PluginContext::new(PluginEventBus::new());
        let registration = ctx.register_owned_service("answer", 41i32);

        assert_eq!(registration.name(), "answer");
        assert_eq!(*ctx.get_service::<i32>("answer").unwrap(), 41);

        ctx.register_service("answer", 42i32);
        drop(registration);

        assert_eq!(*ctx.get_service::<i32>("answer").unwrap(), 42);
    }

    #[test]
    fn test_owned_service_is_removed_and_dropped_outside_store_lock() {
        let ctx = PluginContext::new(PluginEventBus::new());
        let dropped_without_lock = Arc::new(AtomicBool::new(false));
        let registration = ctx.register_owned_service(
            "owned_probe",
            ServiceDropProbe {
                services: Arc::downgrade(&ctx.services),
                dropped_without_lock: Arc::clone(&dropped_without_lock),
            },
        );

        drop(registration);

        assert!(ctx.get_service::<ServiceDropProbe>("owned_probe").is_none());
        assert!(dropped_without_lock.load(Ordering::SeqCst));
    }

    #[test]
    fn test_owned_service_drop_recovers_poisoned_store() {
        let ctx = PluginContext::new(PluginEventBus::new());
        let registration = ctx.register_owned_service("owned_answer", 42i32);
        let services = Arc::clone(&ctx.services);
        let _ = thread::spawn(move || {
            let _guard = services.lock().unwrap();
            panic!("poison owned service store");
        })
        .join();

        drop(registration);

        assert!(!ctx.services.is_poisoned());
        assert!(ctx.get_service::<i32>("owned_answer").is_none());
    }

    #[test]
    fn test_context_recovers_poisoned_stores() {
        let ctx = PluginContext::new(PluginEventBus::new());
        let config = Arc::clone(&ctx.config);
        let services = Arc::clone(&ctx.services);

        let _ = thread::spawn(move || {
            let _guard = config.lock().unwrap();
            panic!("poison config");
        })
        .join();
        let _ = thread::spawn(move || {
            let _guard = services.lock().unwrap();
            panic!("poison services");
        })
        .join();

        ctx.set_config("recovered", "yes");
        ctx.register_service("answer", 42i32);

        assert_eq!(ctx.get_config("recovered"), Some("yes".to_string()));
        assert_eq!(*ctx.get_service::<i32>("answer").unwrap(), 42);
        assert!(!ctx.config.is_poisoned());
        assert!(!ctx.services.is_poisoned());
    }

    #[test]
    fn test_input_conversion_happens_before_config_lock() {
        let ctx = PluginContext::new(PluginEventBus::new());

        let result = catch_unwind(AssertUnwindSafe(|| {
            ctx.set_config(PanickingString, "value");
        }));

        assert!(result.is_err());
        assert!(!ctx.config.is_poisoned());
        ctx.set_config("healthy", "value");
        assert_eq!(ctx.get_config("healthy"), Some("value".to_string()));
    }

    #[test]
    fn test_service_is_dropped_after_store_lock_is_released() {
        let ctx = PluginContext::new(PluginEventBus::new());
        let dropped_without_lock = Arc::new(AtomicBool::new(false));
        ctx.register_service(
            "probe",
            ServiceDropProbe {
                services: Arc::downgrade(&ctx.services),
                dropped_without_lock: Arc::clone(&dropped_without_lock),
            },
        );

        assert!(ctx.unregister_service("probe"));
        assert!(dropped_without_lock.load(Ordering::SeqCst));
    }
}
