// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use std::sync::{Mutex, MutexGuard};

pub(super) fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let guard = poisoned.into_inner();
            mutex.clear_poison();
            guard
        }
    }
}

pub mod cpu_dense;
pub mod faer_dense;

#[cfg(feature = "wgpu_frac")]
pub mod wgpu_frac;

#[cfg(feature = "wgpu_dense")]
pub mod wgpu_dense;

#[cfg(feature = "wgpu_dense")]
pub mod wgpu_util;

#[cfg(feature = "hip")]
pub mod hip_dense;
