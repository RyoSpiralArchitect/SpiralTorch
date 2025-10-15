// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Engine/Optimizer hook points (registration + no-op safe calls)
use std::sync::{Mutex, OnceLock};

type GradHookFn = fn(grad: &mut [f32], world: u32);
type PartHookFn = fn(params: &[f32], rank: u32, world: u32) -> Vec<f32>;

static GRAD_HOOK: OnceLock<Mutex<Option<GradHookFn>>> = OnceLock::new();
static PART_HOOK: OnceLock<Mutex<Option<PartHookFn>>> = OnceLock::new();

pub fn register_onebit_allreduce(h: GradHookFn) {
    GRAD_HOOK
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .replace(h);
}
pub fn register_zero_partition(h: PartHookFn) {
    PART_HOOK
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .replace(h);
}

pub fn call_onebit_allreduce(grad: &mut [f32], world: u32) {
    if let Some(guard) = GRAD_HOOK.get() {
        if let Some(h) = *guard.lock().unwrap() {
            (h)(grad, world);
            return;
        }
    }
    // no-op
}
pub fn call_zero_partition(params: &[f32], rank: u32, world: u32) -> Vec<f32> {
    if let Some(guard) = PART_HOOK.get() {
        if let Some(h) = *guard.lock().unwrap() {
            return (h)(params, rank, world);
        }
    }
    params.to_vec()
}
