// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Lightweight hooks for observing tensor operations without introducing a
//! dependency on the higher-level plugin/event system.

use serde_json::Value;
use std::cell::{Cell, RefCell};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock, RwLock};

/// Metadata about a completed tensor operation.
#[derive(Clone, Debug)]
pub struct TensorOpEvent {
    pub op_name: &'static str,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

/// Observer callback invoked after a tensor operation completes.
pub type TensorOpObserver = Arc<dyn Fn(&TensorOpEvent) + Send + Sync + 'static>;

static TENSOR_OP_OBSERVER: OnceLock<RwLock<Option<TensorOpObserver>>> = OnceLock::new();

/// Metadata about a tensor operation decision (backend choice, fused kernel variant, etc).
#[derive(Clone, Debug)]
pub struct TensorOpMetaEvent {
    pub op_name: &'static str,
    pub data: Value,
}

/// Observer callback invoked after a tensor operation emits metadata.
pub type TensorOpMetaObserver = Arc<dyn Fn(&TensorOpMetaEvent) + Send + Sync + 'static>;

static TENSOR_OP_META_OBSERVER: OnceLock<RwLock<Option<TensorOpMetaObserver>>> = OnceLock::new();

thread_local! {
    static IN_OBSERVER_CALLBACK: Cell<bool> = const { Cell::new(false) };
    static IN_META_OBSERVER_CALLBACK: Cell<bool> = const { Cell::new(false) };
    static THREAD_TENSOR_OP_META_OBSERVER: RefCell<Option<TensorOpMetaObserver>> =
        const { RefCell::new(None) };
}

struct MetaObserverCallbackReset;

impl Drop for MetaObserverCallbackReset {
    fn drop(&mut self) {
        IN_META_OBSERVER_CALLBACK.with(|flag| flag.set(false));
    }
}

/// Install (or clear) the global tensor operation observer.
///
/// Returns the previously installed observer, if any.
pub fn set_tensor_op_observer(observer: Option<TensorOpObserver>) -> Option<TensorOpObserver> {
    let lock = TENSOR_OP_OBSERVER.get_or_init(|| RwLock::new(None));
    let mut slot = lock
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::mem::replace(&mut *slot, observer)
}

/// Install (or clear) the global tensor operation metadata observer.
///
/// This process-wide slot is intended for long-lived integrations such as the
/// plugin event bridge. Temporary captures and tests should prefer
/// [`set_thread_meta_observer`] so concurrent callers cannot replace each
/// other's callback.
///
/// Returns the previously installed observer, if any.
pub fn set_tensor_op_meta_observer(
    observer: Option<TensorOpMetaObserver>,
) -> Option<TensorOpMetaObserver> {
    let lock = TENSOR_OP_META_OBSERVER.get_or_init(|| RwLock::new(None));
    let mut slot = lock
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    std::mem::replace(&mut *slot, observer)
}

/// Install (or clear) a tensor operation metadata observer for the current thread.
///
/// Thread observers receive events before the process-global observer and are
/// useful for isolating concurrent training, diagnostic sessions, and tests.
/// Returns the previously installed observer for this thread so callers can
/// restore it.
pub fn set_thread_meta_observer(
    observer: Option<TensorOpMetaObserver>,
) -> Option<TensorOpMetaObserver> {
    THREAD_TENSOR_OP_META_OBSERVER.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), observer))
}

/// Returns whether a tensor operation metadata observer is currently installed.
///
/// This lets hot paths skip expensive metadata precomputation while preserving
/// the allocation-free default path used by [`emit_tensor_op_meta`].
pub fn tensor_op_meta_observer_installed() -> bool {
    if THREAD_TENSOR_OP_META_OBSERVER.with(|slot| slot.borrow().is_some()) {
        return true;
    }
    TENSOR_OP_META_OBSERVER
        .get()
        .and_then(|lock| {
            lock.read()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .as_ref()
                .map(|_| ())
        })
        .is_some()
}

/// Emit an operation event to the currently installed observer.
///
/// This is a no-op unless an observer has been registered via
/// [`set_tensor_op_observer`]. When no observer is present this function does
/// not allocate.
pub fn emit_tensor_op(op_name: &'static str, input_shape: &[usize], output_shape: &[usize]) {
    let lock = match TENSOR_OP_OBSERVER.get() {
        Some(lock) => lock,
        None => return,
    };
    let observer = lock
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    let Some(observer) = observer else {
        return;
    };

    let already_in_callback = IN_OBSERVER_CALLBACK.with(|flag| {
        if flag.get() {
            true
        } else {
            flag.set(true);
            false
        }
    });
    if already_in_callback {
        return;
    }

    let event = TensorOpEvent {
        op_name,
        input_shape: input_shape.to_vec(),
        output_shape: output_shape.to_vec(),
    };

    let _ = catch_unwind(AssertUnwindSafe(|| observer(&event)));

    IN_OBSERVER_CALLBACK.with(|flag| flag.set(false));
}

/// Emit an operation metadata event to the currently installed observer.
///
/// The `build` closure is only evaluated when an observer is present, keeping
/// the hot path allocation-free by default.
pub fn emit_tensor_op_meta<F>(op_name: &'static str, build: F)
where
    F: FnOnce() -> Value,
{
    let thread_observer =
        THREAD_TENSOR_OP_META_OBSERVER.with(|slot| slot.borrow().as_ref().cloned());
    let global_observer = TENSOR_OP_META_OBSERVER.get().and_then(|lock| {
        lock.read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .as_ref()
            .cloned()
    });
    if thread_observer.is_none() && global_observer.is_none() {
        return;
    }

    let already_in_callback = IN_META_OBSERVER_CALLBACK.with(|flag| {
        if flag.get() {
            true
        } else {
            flag.set(true);
            false
        }
    });
    if already_in_callback {
        return;
    }
    let _callback_reset = MetaObserverCallbackReset;

    let event = TensorOpMetaEvent {
        op_name,
        data: build(),
    };

    if let Some(observer) = thread_observer.as_ref() {
        let _ = catch_unwind(AssertUnwindSafe(|| observer(&event)));
    }
    if let Some(observer) = global_observer.as_ref() {
        let duplicates_thread_observer = thread_observer
            .as_ref()
            .map(|thread| Arc::ptr_eq(thread, observer))
            .unwrap_or(false);
        if !duplicates_thread_observer {
            let _ = catch_unwind(AssertUnwindSafe(|| observer(&event)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn thread_meta_observer_does_not_capture_other_threads() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = set_thread_meta_observer(Some(Arc::new(move |event| {
            captured.lock().unwrap().push(event.op_name);
        })));

        emit_tensor_op_meta("thread_local_event", || Value::Null);
        std::thread::spawn(|| emit_tensor_op_meta("other_thread_event", || Value::Null))
            .join()
            .unwrap();
        set_thread_meta_observer(previous);

        assert_eq!(*events.lock().unwrap(), vec!["thread_local_event"]);
    }

    #[test]
    fn thread_and_global_meta_observers_both_receive_local_events() {
        let thread_events = Arc::new(Mutex::new(Vec::new()));
        let thread_capture = Arc::clone(&thread_events);
        let previous_thread = set_thread_meta_observer(Some(Arc::new(move |event| {
            thread_capture.lock().unwrap().push(event.op_name);
        })));
        let global_events = Arc::new(Mutex::new(Vec::new()));
        let global_capture = Arc::clone(&global_events);
        let previous_global = set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            global_capture.lock().unwrap().push(event.op_name);
        })));

        emit_tensor_op_meta("thread_and_global_event", || Value::Null);
        set_thread_meta_observer(previous_thread);
        set_tensor_op_meta_observer(previous_global);

        assert!(thread_events
            .lock()
            .unwrap()
            .contains(&"thread_and_global_event"));
        assert!(global_events
            .lock()
            .unwrap()
            .contains(&"thread_and_global_event"));
    }

    #[test]
    fn concurrent_thread_meta_observers_remain_isolated() {
        let run = |op_name| {
            std::thread::spawn(move || {
                let events = Arc::new(Mutex::new(Vec::new()));
                let captured = Arc::clone(&events);
                let previous = set_thread_meta_observer(Some(Arc::new(move |event| {
                    captured.lock().unwrap().push(event.op_name);
                })));
                emit_tensor_op_meta(op_name, || Value::Null);
                set_thread_meta_observer(previous);
                Arc::try_unwrap(events).unwrap().into_inner().unwrap()
            })
        };

        let left = run("left_thread_event");
        let right = run("right_thread_event");
        assert_eq!(left.join().unwrap(), vec!["left_thread_event"]);
        assert_eq!(right.join().unwrap(), vec!["right_thread_event"]);
    }

    #[test]
    fn metadata_builder_panic_does_not_stick_reentrancy_guard() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&events);
        let previous = set_thread_meta_observer(Some(Arc::new(move |event| {
            captured.lock().unwrap().push(event.op_name);
        })));

        let panic = std::panic::catch_unwind(|| {
            emit_tensor_op_meta("panicking_metadata", || panic!("metadata build panic"));
        });
        assert!(panic.is_err());
        emit_tensor_op_meta("metadata_after_panic", || Value::Null);
        set_thread_meta_observer(previous);

        assert_eq!(*events.lock().unwrap(), vec!["metadata_after_panic"]);
    }
}
