// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Thread-local application of the execution plan owned by `st-core`.

use serde_json::json;
use st_tensor::{
    AttentionBackend, LayerNormBackend, MatmulBackend, SoftmaxBackend, TensorUtilBackend,
};
use std::cell::RefCell;

pub use super::execution_plan::{
    evaluate_runtime_execution_plan, AcceleratorFallback, BackendPolicy, ExecutionConfig,
    RuntimeComponentRoute, RuntimeComponentRouteClass, RuntimeExecutionComponent,
    RuntimeExecutionPlanError, RuntimeExecutionPlanPayload, RuntimeExecutionPlanRequest,
    RuntimeExecutionPlanStatus, RuntimeTensorBackend, RuntimeTensorBackendPolicy, TensorUtilRoute,
    TensorUtilRouteStatus,
};

thread_local! {
    static ACTIVE_BACKEND_POLICY: RefCell<Option<BackendPolicy>> = const { RefCell::new(None) };
}

/// RAII guard that restores the previous thread-local backend policy.
#[derive(Debug)]
pub struct BackendPolicyGuard {
    previous: Option<BackendPolicy>,
    _tensor_fallback_guard: st_tensor::execution::AcceleratorFallbackGuard,
}

impl Drop for BackendPolicyGuard {
    fn drop(&mut self) {
        ACTIVE_BACKEND_POLICY.with(|slot| {
            *slot.borrow_mut() = self.previous.take();
        });
    }
}

/// Installs `policy` for the current thread until the returned guard is dropped.
pub fn push_backend_policy(policy: BackendPolicy) -> BackendPolicyGuard {
    let tensor_fallback_guard = st_tensor::execution::push_accelerator_fallback(
        policy.execution_config().accelerator_fallback,
    );
    let previous = ACTIVE_BACKEND_POLICY.with(|slot| slot.replace(Some(policy)));
    BackendPolicyGuard {
        previous,
        _tensor_fallback_guard: tensor_fallback_guard,
    }
}

/// Validates and installs a committed Rust-owned runtime execution plan.
pub fn push_runtime_execution_plan(
    plan: &RuntimeExecutionPlanPayload,
) -> Result<BackendPolicyGuard, RuntimeExecutionPlanError> {
    let policy = BackendPolicy::try_from_runtime_plan(plan)?;
    Ok(push_backend_policy(policy))
}

/// Returns the current thread-local backend policy, if any.
pub fn current_backend_policy() -> Option<BackendPolicy> {
    ACTIVE_BACKEND_POLICY.with(|slot| *slot.borrow())
}

/// Returns the captured accelerator fallback contract, preserving legacy env behavior outside a policy scope.
pub fn current_accelerator_fallback() -> AcceleratorFallback {
    st_tensor::execution::current_accelerator_fallback()
}

/// Returns the current dense matmul backend, falling back to tensor-level Auto.
pub fn current_matmul_backend() -> MatmulBackend {
    current_backend_policy()
        .map(BackendPolicy::matmul_backend)
        .unwrap_or(MatmulBackend::Auto)
}

/// Returns the current prepacked matmul backend, falling back to tensor-level Auto.
pub fn current_prepacked_matmul_backend() -> MatmulBackend {
    current_backend_policy()
        .map(BackendPolicy::prepacked_matmul_backend)
        .unwrap_or(MatmulBackend::Auto)
}

/// Returns the current layer-normalisation backend, falling back to tensor-level Auto.
pub fn current_layer_norm_backend() -> LayerNormBackend {
    current_backend_policy()
        .map(BackendPolicy::layer_norm_backend)
        .unwrap_or(LayerNormBackend::Auto)
}

/// Returns the current fused-attention backend, falling back to tensor-level Auto.
pub fn current_attention_backend() -> AttentionBackend {
    current_backend_policy()
        .map(BackendPolicy::attention_backend)
        .unwrap_or(AttentionBackend::Auto)
}

/// Returns the current row-wise softmax backend, falling back to tensor-level Auto.
pub fn current_softmax_backend() -> SoftmaxBackend {
    current_backend_policy()
        .map(BackendPolicy::softmax_backend)
        .unwrap_or(SoftmaxBackend::Auto)
}

/// Returns the current tensor utility backend, falling back to legacy Auto.
pub fn current_tensor_util_backend() -> TensorUtilBackend {
    current_backend_policy()
        .map(BackendPolicy::tensor_util_backend)
        .unwrap_or(TensorUtilBackend::Auto)
}

/// Resolves the complete tensor utility route for an operation of `values` elements.
pub fn current_tensor_util_route(values: usize) -> TensorUtilRoute {
    let Some(policy) = current_backend_policy() else {
        return TensorUtilRoute {
            requested_backend: TensorUtilBackend::Auto,
            selected_backend: TensorUtilBackend::Auto,
            values,
            threshold: 0,
            status: TensorUtilRouteStatus::Direct,
        };
    };
    let route = policy.tensor_util_route(values);
    if route.records_threshold_decision() {
        emit_tensor_util_route(policy, route);
    }
    route
}

/// Applies the core execution plan to a tensor utility operation of `values` elements.
pub fn current_tensor_util_backend_for_values(values: usize) -> TensorUtilBackend {
    current_tensor_util_route(values).selected_backend
}

fn emit_tensor_util_route(policy: BackendPolicy, route: TensorUtilRoute) {
    st_tensor::emit_tensor_op_meta("tensor_util_route", || {
        let mut payload = json!({
            "requested_backend": route.requested_backend_label(),
            "selected_backend": route.selected_backend_label(),
            "status": route.status.as_str(),
            "choice_source": "threshold_guard",
            "semantic_owner": "st-core",
            "accelerator_fallback": policy
                .execution_config()
                .accelerator_fallback
                .as_str(),
            "values": route.values,
            "threshold": route.threshold,
        });
        if let Some(commitment) = policy.runtime_plan_output_sha256_hex() {
            payload["execution_plan_output_sha256"] = commitment.into();
            payload["execution_plan_committed"] = true.into();
        }
        payload
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::device_caps::DeviceCaps;
    use crate::backend::runtime_probe::{evaluate_runtime_device_probe, RuntimeDeviceProbeRequest};
    use std::sync::{Arc, Mutex};

    #[test]
    fn backend_policy_guard_restores_previous_policy() {
        assert!(current_backend_policy().is_none());
        let outer = BackendPolicy::explicit(
            DeviceCaps::cpu(),
            MatmulBackend::CpuNaive,
            MatmulBackend::CpuNaive,
            LayerNormBackend::Cpu,
            AttentionBackend::Cpu,
            SoftmaxBackend::Cpu,
        );
        let inner = BackendPolicy::explicit(
            DeviceCaps::cpu(),
            MatmulBackend::CpuFaer,
            MatmulBackend::CpuFaer,
            LayerNormBackend::Auto,
            AttentionBackend::Auto,
            SoftmaxBackend::Auto,
        );

        let outer_guard = push_backend_policy(outer);
        assert_eq!(current_matmul_backend(), MatmulBackend::CpuNaive);
        {
            let _inner_guard = push_backend_policy(inner);
            assert_eq!(current_matmul_backend(), MatmulBackend::CpuFaer);
            assert_eq!(current_prepacked_matmul_backend(), MatmulBackend::CpuFaer);
            assert_eq!(current_layer_norm_backend(), LayerNormBackend::Auto);
            assert_eq!(current_attention_backend(), AttentionBackend::Auto);
            assert_eq!(current_softmax_backend(), SoftmaxBackend::Auto);
        }
        assert_eq!(current_matmul_backend(), MatmulBackend::CpuNaive);
        assert_eq!(current_layer_norm_backend(), LayerNormBackend::Cpu);
        assert_eq!(current_attention_backend(), AttentionBackend::Cpu);
        assert_eq!(current_softmax_backend(), SoftmaxBackend::Cpu);
        drop(outer_guard);
        assert!(current_backend_policy().is_none());
    }

    #[test]
    fn active_policy_owns_the_fallback_contract() {
        let strict = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::cpu(),
            ExecutionConfig::new(AcceleratorFallback::Forbid, 1024),
        );
        let _guard = push_backend_policy(strict);
        assert_eq!(current_accelerator_fallback(), AcceleratorFallback::Forbid);
        assert_eq!(
            st_tensor::execution::current_accelerator_fallback(),
            AcceleratorFallback::Forbid
        );
    }

    #[test]
    fn tensor_utility_threshold_route_emits_the_core_decision() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous_observer = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::wgpu(32, true, 256),
            ExecutionConfig::new(AcceleratorFallback::Allow, 1024),
        );
        let guard = push_backend_policy(policy);
        let route = current_tensor_util_route(8);
        drop(guard);

        st_tensor::set_thread_meta_observer(previous_observer);

        assert_eq!(route.requested_backend, TensorUtilBackend::GpuWgpu);
        assert_eq!(route.selected_backend, TensorUtilBackend::Cpu);
        assert_eq!(route.status, TensorUtilRouteStatus::CpuThreshold);
        let events = events.lock().unwrap();
        let (_, data) = events
            .iter()
            .find(|(op_name, _)| *op_name == "tensor_util_route")
            .expect("tensor util route metadata");
        assert_eq!(data["requested_backend"], "wgpu");
        assert_eq!(data["selected_backend"], "cpu");
        assert_eq!(data["status"], "cpu_threshold");
        assert_eq!(data["semantic_owner"], "st-core");
        assert_eq!(data["values"], 8);
        assert_eq!(data["threshold"], 1024);
        assert!(data.get("backend").is_none());
    }

    #[test]
    fn tensor_utility_route_without_a_policy_preserves_legacy_auto() {
        assert!(current_backend_policy().is_none());

        let route = current_tensor_util_route(37);

        assert_eq!(route.requested_backend, TensorUtilBackend::Auto);
        assert_eq!(route.selected_backend, TensorUtilBackend::Auto);
        assert_eq!(route.status, TensorUtilRouteStatus::Direct);
        assert_eq!(route.values, 37);
        assert_eq!(route.threshold, 0);
    }

    #[test]
    fn committed_runtime_plan_is_the_executable_policy_entrypoint() {
        let probe = evaluate_runtime_device_probe(RuntimeDeviceProbeRequest {
            requested_backend: crate::backend::device_caps::BackendKind::Cpu,
            caps: DeviceCaps::cpu(),
            mps_probe: None,
            requested_workgroup: None,
            cols: None,
            tile_hint: None,
            compaction_hint: None,
        })
        .expect("CPU probe");
        let plan = evaluate_runtime_execution_plan(RuntimeExecutionPlanRequest {
            runtime_probe: probe,
            execution_config: ExecutionConfig::default(),
            tensor_util_values: None,
            required_native_components: vec![RuntimeExecutionComponent::DenseMatmul],
        })
        .expect("CPU execution plan");

        let guard = push_runtime_execution_plan(&plan).expect("install committed plan");
        let current = current_backend_policy().expect("active policy");
        assert_eq!(current_matmul_backend(), MatmulBackend::CpuFaer);
        assert_eq!(
            current.runtime_plan_output_sha256_hex().as_deref(),
            Some(plan.output_sha256.as_str())
        );
        drop(guard);
        assert!(current_backend_policy().is_none());
    }
}
