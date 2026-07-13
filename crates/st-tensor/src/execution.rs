// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Tensor-level application of an already resolved execution contract.

use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

pub use spiral_config::execution::AcceleratorFallback;

thread_local! {
    static ACTIVE_ACCELERATOR_FALLBACK: Cell<Option<AcceleratorFallback>> = const { Cell::new(None) };
}

/// RAII guard that restores the previous tensor fallback contract.
#[derive(Debug)]
pub struct AcceleratorFallbackGuard {
    previous: Option<AcceleratorFallback>,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for AcceleratorFallbackGuard {
    fn drop(&mut self) {
        ACTIVE_ACCELERATOR_FALLBACK.with(|slot| slot.set(self.previous));
    }
}

/// Installs a resolved fallback contract for tensor operations on this thread.
pub fn push_accelerator_fallback(fallback: AcceleratorFallback) -> AcceleratorFallbackGuard {
    let previous = ACTIVE_ACCELERATOR_FALLBACK.with(|slot| slot.replace(Some(fallback)));
    AcceleratorFallbackGuard {
        previous,
        _not_send: PhantomData,
    }
}

/// Returns the active contract, preserving direct-call compatibility outside a policy scope.
pub fn current_accelerator_fallback() -> AcceleratorFallback {
    ACTIVE_ACCELERATOR_FALLBACK
        .with(Cell::get)
        .unwrap_or_else(AcceleratorFallback::from_env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_guard_restores_nested_contracts() {
        let outer = push_accelerator_fallback(AcceleratorFallback::Forbid);
        assert_eq!(current_accelerator_fallback(), AcceleratorFallback::Forbid);

        {
            let _inner = push_accelerator_fallback(AcceleratorFallback::Allow);
            assert_eq!(current_accelerator_fallback(), AcceleratorFallback::Allow);
        }

        assert_eq!(current_accelerator_fallback(), AcceleratorFallback::Forbid);
        drop(outer);
    }
}
