// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Event system for inter-plugin communication.

use std::any::Any;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Events that can be emitted and handled by plugins.
#[derive(Debug, Clone)]
pub enum PluginEvent {
    /// System initialization event
    SystemInit,
    /// System shutdown event
    SystemShutdown,
    /// A plugin was loaded
    PluginLoaded { plugin_id: String },
    /// A plugin was unloaded
    PluginUnloaded { plugin_id: String },
    /// Tensor operation completed
    TensorOp {
        op_name: String,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
    },
    /// Training epoch started
    EpochStart { epoch: usize },
    /// Training epoch completed
    EpochEnd { epoch: usize, loss: f32 },
    /// Backend changed
    BackendChanged { backend: String },
    /// Telemetry data available
    Telemetry { data: HashMap<String, f32> },
    /// Custom event with arbitrary payload
    Custom {
        event_type: String,
        data: Arc<dyn Any + Send + Sync>,
    },
}

impl PluginEvent {
    /// Create a custom event with a typed payload.
    pub fn custom<T: Any + Send + Sync + 'static>(event_type: impl Into<String>, data: T) -> Self {
        Self::Custom {
            event_type: event_type.into(),
            data: Arc::new(data),
        }
    }

    /// Try to extract a typed payload from a custom event.
    pub fn downcast_data<T: Any + Send + Sync>(&self) -> Option<&T> {
        match self {
            Self::Custom { data, .. } => data.downcast_ref::<T>(),
            _ => None,
        }
    }
}

/// Listener callback for plugin events.
pub type EventListener = Arc<dyn Fn(&PluginEvent) + Send + Sync>;

/// Event bus for publishing and subscribing to plugin events.
#[derive(Clone)]
pub struct PluginEventBus {
    listeners: Arc<Mutex<HashMap<String, Vec<(usize, EventListener)>>>>,
    next_id: Arc<AtomicUsize>,
}

impl PluginEventBus {
    /// Create a new event bus.
    pub fn new() -> Self {
        Self {
            listeners: Arc::new(Mutex::new(HashMap::new())),
            next_id: Arc::new(AtomicUsize::new(1)),
        }
    }

    /// Subscribe to events of a specific type.
    ///
    /// The `event_type` is matched against the variant name of PluginEvent.
    /// Use "*" to subscribe to all events.
    pub fn subscribe(&self, event_type: impl Into<String>, listener: EventListener) -> usize {
        let event_type = event_type.into();
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut listeners = self.listeners.lock().unwrap();
        listeners.entry(event_type).or_default().push((id, listener));
        id
    }

    /// Unsubscribe a previously registered listener.
    ///
    /// Returns `true` when a subscription was found and removed.
    pub fn unsubscribe(&self, event_type: &str, id: usize) -> bool {
        let mut listeners = self.listeners.lock().unwrap();
        let Some(bucket) = listeners.get_mut(event_type) else {
            return false;
        };
        let before = bucket.len();
        bucket.retain(|(existing, _)| *existing != id);
        let removed = bucket.len() != before;
        if bucket.is_empty() {
            listeners.remove(event_type);
        }
        removed
    }

    /// Returns true when there are listeners registered for the provided event type
    /// or for the wildcard `"*"`.
    pub fn has_listeners(&self, event_type: &str) -> bool {
        let listeners = self.listeners.lock().unwrap();
        listeners
            .get(event_type)
            .is_some_and(|bucket| !bucket.is_empty())
            || listeners
                .get("*")
                .is_some_and(|bucket| !bucket.is_empty())
    }

    /// Publish an event to all interested listeners.
    pub fn publish(&self, event: &PluginEvent) {
        let event_type = self.event_type_name(event);

        let listeners: Vec<EventListener> = {
            let listeners = self.listeners.lock().unwrap();
            let mut collected = Vec::new();
            if let Some(specific_listeners) = listeners.get(&event_type) {
                collected.extend(specific_listeners.iter().map(|(_, listener)| listener.clone()));
            }
            if let Some(wildcard_listeners) = listeners.get("*") {
                collected.extend(wildcard_listeners.iter().map(|(_, listener)| listener.clone()));
            }
            collected
        };

        for listener in listeners {
            listener(event);
        }
    }

    /// Get the event type name for matching.
    fn event_type_name(&self, event: &PluginEvent) -> String {
        match event {
            PluginEvent::SystemInit => "SystemInit".to_string(),
            PluginEvent::SystemShutdown => "SystemShutdown".to_string(),
            PluginEvent::PluginLoaded { .. } => "PluginLoaded".to_string(),
            PluginEvent::PluginUnloaded { .. } => "PluginUnloaded".to_string(),
            PluginEvent::TensorOp { .. } => "TensorOp".to_string(),
            PluginEvent::EpochStart { .. } => "EpochStart".to_string(),
            PluginEvent::EpochEnd { .. } => "EpochEnd".to_string(),
            PluginEvent::BackendChanged { .. } => "BackendChanged".to_string(),
            PluginEvent::Telemetry { .. } => "Telemetry".to_string(),
            PluginEvent::Custom { event_type, .. } => event_type.clone(),
        }
    }
}

impl Default for PluginEventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_event_bus_subscription() {
        let bus = PluginEventBus::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let id = bus.subscribe("EpochStart", Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        bus.publish(&PluginEvent::EpochStart { epoch: 1 });
        bus.publish(&PluginEvent::EpochStart { epoch: 2 });

        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert!(bus.unsubscribe("EpochStart", id));
        bus.publish(&PluginEvent::EpochStart { epoch: 3 });
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_wildcard_subscription() {
        let bus = PluginEventBus::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        bus.subscribe("*", Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        }));

        bus.publish(&PluginEvent::SystemInit);
        bus.publish(&PluginEvent::EpochStart { epoch: 1 });
        bus.publish(&PluginEvent::SystemShutdown);

        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_custom_event() {
        #[derive(Debug, Clone, PartialEq)]
        struct CustomData {
            value: i32,
        }

        let event = PluginEvent::custom("MyEvent", CustomData { value: 42 });
        let data = event.downcast_data::<CustomData>().unwrap();
        assert_eq!(data.value, 42);
    }
}
