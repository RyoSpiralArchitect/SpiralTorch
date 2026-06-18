// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Engine/Optimizer hook points (registration + no-op safe calls)
use std::sync::{Mutex, OnceLock};

use st_tensor::{emit_tensor_op, emit_tensor_op_meta, tensor_op_meta_observer_installed};

type GradHookFn = fn(grad: &mut [f32], world: u32);
type PartHookFn = fn(params: &[f32], rank: u32, world: u32) -> Vec<f32>;

static GRAD_HOOK: OnceLock<Mutex<Option<GradHookFn>>> = OnceLock::new();
static PART_HOOK: OnceLock<Mutex<Option<PartHookFn>>> = OnceLock::new();

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

fn l2_norm(values: &[f32]) -> f32 {
    values
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
}

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
    let capture_meta = tensor_op_meta_observer_installed();
    let before_l2 = if capture_meta { l2_norm(grad) } else { 0.0 };
    let mut registered = false;
    if let Some(guard) = GRAD_HOOK.get() {
        if let Some(h) = *guard.lock().unwrap() {
            registered = true;
            (h)(grad, world);
        }
    }
    emit_tensor_op("onebit_allreduce_hook", &[grad.len()], &[grad.len()]);
    if capture_meta {
        let after_l2 = l2_norm(grad);
        emit_tensor_op_meta("onebit_allreduce_hook", || {
            serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "kind": "st_core_onebit_allreduce_hook",
                "hook_dispatch_backend": "cpu_callback",
                "gradient_mutation_backend": if registered { "registered_host_callback" } else { "noop_cpu" },
                "hook_mode": "opaque_in_place_gradient_callback",
                "route_blocker": "opaque_registered_host_callback",
                "registered": registered,
                "world": world,
                "gradient_len": grad.len(),
                "gradient_l2_before": before_l2,
                "gradient_l2_after": after_l2,
                "gradient_l2_ratio": if before_l2 > f32::EPSILON {
                    after_l2 / before_l2
                } else {
                    1.0
                },
                "estimated_gradient_values": grad.len(),
            })
        });
    }
    // no-op
}

pub fn call_zero_partition(params: &[f32], rank: u32, world: u32) -> Vec<f32> {
    let capture_meta = tensor_op_meta_observer_installed();
    let input_l2 = if capture_meta { l2_norm(params) } else { 0.0 };
    if let Some(guard) = PART_HOOK.get() {
        if let Some(h) = *guard.lock().unwrap() {
            let output = (h)(params, rank, world);
            emit_tensor_op("zero_partition_hook", &[params.len()], &[output.len()]);
            if capture_meta {
                let output_l2 = l2_norm(&output);
                emit_tensor_op_meta("zero_partition_hook", || {
                    serde_json::json!({
                        "backend": "cpu",
                        "requested_backend": "auto",
                        "kind": "st_core_zero_partition_hook",
                        "hook_dispatch_backend": "cpu_callback",
                        "partition_backend": "registered_host_callback",
                        "hook_mode": "opaque_parameter_partition_callback",
                        "route_blocker": "opaque_registered_host_callback",
                        "registered": true,
                        "rank": rank,
                        "world": world,
                        "param_len": params.len(),
                        "partition_len": output.len(),
                        "input_l2": finite_meta_f32(input_l2),
                        "output_l2": finite_meta_f32(output_l2),
                        "estimated_parameter_values": params.len(),
                        "estimated_partition_values": output.len(),
                    })
                });
            }
            return output;
        }
    }
    let output = params.to_vec();
    emit_tensor_op("zero_partition_hook", &[params.len()], &[output.len()]);
    if capture_meta {
        let output_l2 = l2_norm(&output);
        emit_tensor_op_meta("zero_partition_hook", || {
            serde_json::json!({
                "backend": "cpu",
                "requested_backend": "auto",
                "kind": "st_core_zero_partition_hook",
                "hook_dispatch_backend": "cpu_callback",
                "partition_backend": "noop_cpu_clone",
                "hook_mode": "identity_parameter_partition",
                "route_blocker": "opaque_registered_host_callback",
                "registered": false,
                "rank": rank,
                "world": world,
                "param_len": params.len(),
                "partition_len": output.len(),
                "input_l2": finite_meta_f32(input_l2),
                "output_l2": finite_meta_f32(output_l2),
                "estimated_parameter_values": params.len(),
                "estimated_partition_values": output.len(),
            })
        });
    }
    output
}

#[cfg(test)]
fn clear_hooks_for_test() {
    if let Some(guard) = GRAD_HOOK.get() {
        *guard.lock().unwrap() = None;
    }
    if let Some(guard) = PART_HOOK.get() {
        *guard.lock().unwrap() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, OnceLock};

    fn hook_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .expect("hook guard")
    }

    fn halve_gradient(grad: &mut [f32], _world: u32) {
        for value in grad {
            *value *= 0.5;
        }
    }

    fn stride_partition(params: &[f32], rank: u32, world: u32) -> Vec<f32> {
        params
            .iter()
            .enumerate()
            .filter(|(idx, _)| (*idx as u32) % world == rank)
            .map(|(_, value)| *value)
            .collect()
    }

    #[test]
    fn hook_calls_emit_backend_meta() {
        let _observer_lock = crate::telemetry::tensor_observer_lock();
        let _hook_guard = hook_guard();
        clear_hooks_for_test();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        register_onebit_allreduce(halve_gradient);
        register_zero_partition(stride_partition);
        let mut grad = vec![2.0, -2.0, 1.0, -1.0];
        call_onebit_allreduce(&mut grad, 2);
        let partition = call_zero_partition(&[1.0, 2.0, 3.0, 4.0], 1, 2);
        st_tensor::set_tensor_op_meta_observer(previous);
        clear_hooks_for_test();

        assert_eq!(grad, vec![1.0, -1.0, 0.5, -0.5]);
        assert_eq!(partition, vec![2.0, 4.0]);
        let events = events.lock().unwrap();
        let allreduce = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "onebit_allreduce_hook"
                    && data["kind"] == "st_core_onebit_allreduce_hook"
                    && data["registered"] == true
            })
            .expect("onebit allreduce metadata event");
        assert_eq!(allreduce.1["world"], 2);
        assert_eq!(allreduce.1["hook_dispatch_backend"], "cpu_callback");
        assert_eq!(
            allreduce.1["gradient_mutation_backend"],
            "registered_host_callback"
        );
        assert_eq!(
            allreduce.1["hook_mode"],
            "opaque_in_place_gradient_callback"
        );
        assert_eq!(
            allreduce.1["route_blocker"],
            "opaque_registered_host_callback"
        );
        assert_eq!(allreduce.1["estimated_gradient_values"], 4);
        assert!(
            allreduce.1["gradient_l2_after"].as_f64().unwrap()
                < allreduce.1["gradient_l2_before"].as_f64().unwrap()
        );

        let zero = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "zero_partition_hook"
                    && data["kind"] == "st_core_zero_partition_hook"
                    && data["registered"] == true
            })
            .expect("zero partition metadata event");
        assert_eq!(zero.1["rank"], 1);
        assert_eq!(zero.1["world"], 2);
        assert_eq!(zero.1["param_len"], 4);
        assert_eq!(zero.1["partition_len"], 2);
        assert_eq!(zero.1["hook_dispatch_backend"], "cpu_callback");
        assert_eq!(zero.1["partition_backend"], "registered_host_callback");
        assert_eq!(zero.1["hook_mode"], "opaque_parameter_partition_callback");
        assert_eq!(zero.1["route_blocker"], "opaque_registered_host_callback");
        assert_eq!(zero.1["estimated_parameter_values"], 4);
        assert_eq!(zero.1["estimated_partition_values"], 2);
    }
}
