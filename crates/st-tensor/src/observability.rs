// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.

//! Lightweight hooks for observing tensor operations without introducing a
//! dependency on the higher-level plugin/event system.

use std::cell::Cell;
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

thread_local! {
    static IN_OBSERVER_CALLBACK: Cell<bool> = Cell::new(false);
}

/// Install (or clear) the global tensor operation observer.
///
/// Returns the previously installed observer, if any.
pub fn set_tensor_op_observer(observer: Option<TensorOpObserver>) -> Option<TensorOpObserver> {
    let lock = TENSOR_OP_OBSERVER.get_or_init(|| RwLock::new(None));
    let mut slot = lock.write().unwrap();
    std::mem::replace(&mut *slot, observer)
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
    let observer = lock.read().unwrap().clone();
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

