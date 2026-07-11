// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Event system for inter-plugin communication.

use std::any::Any;
use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::panic_payload_message;
use super::sync::lock_recover;

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
type ListenerBucket = Vec<(usize, EventListener)>;
type ListenerMap = HashMap<String, ListenerBucket>;

/// A listener that panicked while handling an event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginEventListenerFailure {
    /// Subscription identifier returned by [`PluginEventBus::subscribe`].
    pub subscription_id: usize,
    /// String panic payload, or a stable fallback for non-string payloads.
    pub message: String,
}

/// Delivery summary returned by [`PluginEventBus::publish_report`].
#[must_use = "event dispatch failures should be inspected"]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginEventDispatchReport {
    /// Event type used to select listeners.
    pub event_type: String,
    /// Number of matching specific and wildcard subscriptions.
    pub matched: usize,
    /// Number of listeners that returned without panicking.
    pub delivered: usize,
    /// Per-listener panic details. Remaining listeners are still called.
    pub failures: Vec<PluginEventListenerFailure>,
}

impl PluginEventDispatchReport {
    /// True when every matching listener completed without panicking.
    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }

    /// Number of listeners that panicked.
    pub fn panicked(&self) -> usize {
        self.failures.len()
    }
}

/// Event bus for publishing and subscribing to plugin events.
#[derive(Clone)]
pub struct PluginEventBus {
    listeners: Arc<Mutex<ListenerMap>>,
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
        let mut listeners = lock_recover(&self.listeners);
        listeners
            .entry(event_type)
            .or_default()
            .push((id, listener));
        id
    }

    /// Unsubscribe a previously registered listener.
    ///
    /// Returns `true` when a subscription was found and removed.
    pub fn unsubscribe(&self, event_type: &str, id: usize) -> bool {
        let removed = {
            let mut listeners = lock_recover(&self.listeners);
            let Some(bucket) = listeners.get_mut(event_type) else {
                return false;
            };
            let Some(position) = bucket.iter().position(|(existing, _)| *existing == id) else {
                return false;
            };
            let (_, listener) = bucket.remove(position);
            if bucket.is_empty() {
                listeners.remove(event_type);
            }
            listener
        };
        drop(removed);
        true
    }

    /// Returns true when there are listeners registered for the provided event type
    /// or for the wildcard `"*"`.
    pub fn has_listeners(&self, event_type: &str) -> bool {
        let listeners = lock_recover(&self.listeners);
        listeners
            .get(event_type)
            .is_some_and(|bucket| !bucket.is_empty())
            || listeners.get("*").is_some_and(|bucket| !bucket.is_empty())
    }

    /// Remove all listeners (or those matching an event type).
    ///
    /// Returns the number of removed subscriptions.
    pub fn clear_listeners(&self, event_type: Option<&str>) -> usize {
        let removed: Vec<ListenerBucket> = {
            let mut listeners = lock_recover(&self.listeners);
            match event_type {
                Some(event_type) => listeners.remove(event_type).into_iter().collect(),
                None => listeners.drain().map(|(_, bucket)| bucket).collect(),
            }
        };
        removed.iter().map(Vec::len).sum()
    }

    /// Publish an event to all interested listeners.
    ///
    /// Listener panics are isolated so one extension cannot interrupt later listeners or the
    /// caller. Use [`Self::publish_report`] when the delivery result must be audited.
    pub fn publish(&self, event: &PluginEvent) {
        let _ = self.publish_report(event);
    }

    /// Publish an event and return per-listener delivery diagnostics.
    pub fn publish_report(&self, event: &PluginEvent) -> PluginEventDispatchReport {
        let event_type = self.event_type_name(event);

        let listeners: Vec<(usize, EventListener)> = {
            let listeners = lock_recover(&self.listeners);
            let mut collected = Vec::new();
            if event_type != "*" {
                if let Some(specific_listeners) = listeners.get(&event_type) {
                    collected.extend(
                        specific_listeners
                            .iter()
                            .map(|(id, listener)| (*id, listener.clone())),
                    );
                }
            }
            if let Some(wildcard_listeners) = listeners.get("*") {
                collected.extend(
                    wildcard_listeners
                        .iter()
                        .map(|(id, listener)| (*id, listener.clone())),
                );
            }
            collected
        };

        let matched = listeners.len();
        let mut delivered = 0usize;
        let mut failures = Vec::new();
        for (subscription_id, listener) in listeners {
            match catch_unwind(AssertUnwindSafe(|| listener(event))) {
                Ok(()) => delivered += 1,
                Err(payload) => failures.push(PluginEventListenerFailure {
                    subscription_id,
                    message: panic_payload_message(payload),
                }),
            }
        }

        PluginEventDispatchReport {
            event_type,
            matched,
            delivered,
            failures,
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
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Weak;
    use std::thread;

    struct ListenerDropProbe {
        listeners: Weak<Mutex<ListenerMap>>,
        dropped_without_lock: Arc<AtomicBool>,
    }

    impl Drop for ListenerDropProbe {
        fn drop(&mut self) {
            let unlocked = self
                .listeners
                .upgrade()
                .is_some_and(|listeners| listeners.try_lock().is_ok());
            self.dropped_without_lock.store(unlocked, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_event_bus_subscription() {
        let bus = PluginEventBus::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let id = bus.subscribe(
            "EpochStart",
            Arc::new(move |_| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );

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

        bus.subscribe(
            "*",
            Arc::new(move |_| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );

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

    #[test]
    fn test_clear_listeners() {
        let bus = PluginEventBus::new();
        let noop: EventListener = Arc::new(|_| {});

        let id_epoch = bus.subscribe("EpochStart", noop.clone());
        let id_init = bus.subscribe("SystemInit", noop.clone());
        let id_any = bus.subscribe("*", noop.clone());

        assert_eq!(bus.clear_listeners(Some("EpochStart")), 1);
        assert!(!bus.unsubscribe("EpochStart", id_epoch));

        assert_eq!(bus.clear_listeners(None), 2);
        assert!(!bus.unsubscribe("SystemInit", id_init));
        assert!(!bus.unsubscribe("*", id_any));
        assert!(!bus.has_listeners("EpochStart"));
        assert!(!bus.has_listeners("SystemInit"));

        assert_eq!(bus.clear_listeners(None), 0);
    }

    #[test]
    fn test_event_bus_recovers_poisoned_listener_store() {
        let bus = PluginEventBus::new();
        let listeners = Arc::clone(&bus.listeners);
        let _ = thread::spawn(move || {
            let _guard = listeners.lock().unwrap();
            panic!("poison listeners");
        })
        .join();

        let calls = Arc::new(AtomicUsize::new(0));
        let listener_calls = Arc::clone(&calls);
        bus.subscribe(
            "SystemInit",
            Arc::new(move |_| {
                listener_calls.fetch_add(1, Ordering::SeqCst);
            }),
        );
        bus.publish(&PluginEvent::SystemInit);

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(!bus.listeners.is_poisoned());
    }

    #[test]
    fn test_listener_is_dropped_after_bus_lock_is_released() {
        let bus = PluginEventBus::new();
        let dropped_without_lock = Arc::new(AtomicBool::new(false));
        let probe = ListenerDropProbe {
            listeners: Arc::downgrade(&bus.listeners),
            dropped_without_lock: Arc::clone(&dropped_without_lock),
        };
        let listener: EventListener = Arc::new(move |_| {
            let _ = &probe;
        });
        let id = bus.subscribe("SystemInit", listener);

        assert!(bus.unsubscribe("SystemInit", id));
        assert!(dropped_without_lock.load(Ordering::SeqCst));
    }

    #[test]
    fn test_publish_report_isolates_listener_panics() {
        let bus = PluginEventBus::new();
        let panic_id = bus.subscribe(
            "SystemInit",
            Arc::new(|_| {
                panic!("listener failed");
            }),
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let listener_calls = Arc::clone(&calls);
        bus.subscribe(
            "SystemInit",
            Arc::new(move |_| {
                listener_calls.fetch_add(1, Ordering::SeqCst);
            }),
        );

        let report = bus.publish_report(&PluginEvent::SystemInit);

        assert_eq!(report.event_type, "SystemInit");
        assert_eq!(report.matched, 2);
        assert_eq!(report.delivered, 1);
        assert_eq!(report.panicked(), 1);
        assert!(!report.ok());
        assert_eq!(report.failures[0].subscription_id, panic_id);
        assert_eq!(report.failures[0].message, "listener failed");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        bus.publish(&PluginEvent::SystemInit);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_publish_report_does_not_drop_unknown_panic_payload() {
        struct PanicOnDrop;

        impl Drop for PanicOnDrop {
            fn drop(&mut self) {
                panic!("payload drop panic");
            }
        }

        let bus = PluginEventBus::new();
        bus.subscribe(
            "SystemInit",
            Arc::new(|_| std::panic::panic_any(PanicOnDrop)),
        );

        let report = bus.publish_report(&PluginEvent::SystemInit);

        assert_eq!(report.panicked(), 1);
        assert_eq!(report.failures[0].message, "non-string panic payload");
    }

    #[test]
    fn test_custom_wildcard_event_is_delivered_once() {
        let bus = PluginEventBus::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let listener_calls = Arc::clone(&calls);
        bus.subscribe(
            "*",
            Arc::new(move |_| {
                listener_calls.fetch_add(1, Ordering::SeqCst);
            }),
        );

        let report = bus.publish_report(&PluginEvent::custom("*", ()));

        assert_eq!(report.matched, 1);
        assert_eq!(report.delivered, 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
