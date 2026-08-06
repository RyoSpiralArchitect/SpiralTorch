// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Compatibility re-exports for the execution policy owned by `st-core`.

use crate::plan::RankPlanner;

use st_core::backend::device_caps::DeviceCaps;
pub use st_core::backend::execution::{
    current_accelerator_fallback, current_attention_backend, current_backend_policy,
    current_layer_norm_backend, current_matmul_backend, current_prepacked_matmul_backend,
    current_softmax_backend, current_tensor_util_backend, current_tensor_util_backend_for_values,
    current_tensor_util_route, push_backend_policy, AcceleratorFallback, BackendPolicy,
    BackendPolicyGuard, ExecutionConfig, TensorUtilRoute, TensorUtilRouteStatus,
};
pub use st_core::backend::execution_plan::{
    RuntimeExecutionPlanError, RuntimeExecutionPlanPayload,
};

/// Trainer-local view of the canonical Rust execution contract.
///
/// The context keeps rank planning and tensor execution derived from the same
/// immutable inputs. A committed runtime plan is retained as provenance rather
/// than being reinterpreted by the trainer or a language binding.
#[derive(Clone, Debug)]
pub struct TrainerExecutionContext {
    planner: RankPlanner,
    backend_policy: BackendPolicy,
    runtime_plan: Option<RuntimeExecutionPlanPayload>,
}

impl TrainerExecutionContext {
    /// Builds an uncommitted context for backwards-compatible direct Rust use.
    pub fn from_device_caps_with_config(caps: DeviceCaps, config: ExecutionConfig) -> Self {
        Self {
            planner: RankPlanner::with_execution_config(caps, config),
            backend_policy: BackendPolicy::from_device_caps_with_config(caps, config),
            runtime_plan: None,
        }
    }

    /// Materializes one executable context from a validated committed plan.
    pub fn try_from_runtime_plan(
        plan: &RuntimeExecutionPlanPayload,
    ) -> Result<Self, RuntimeExecutionPlanError> {
        let backend_policy = BackendPolicy::try_from_runtime_plan(plan)?;
        Ok(Self {
            planner: RankPlanner::with_execution_config(
                backend_policy.device_caps(),
                backend_policy.execution_config(),
            ),
            backend_policy,
            runtime_plan: Some(plan.clone()),
        })
    }

    pub const fn planner(&self) -> &RankPlanner {
        &self.planner
    }

    pub const fn backend_policy(&self) -> BackendPolicy {
        self.backend_policy
    }

    pub fn runtime_execution_plan(&self) -> Option<&RuntimeExecutionPlanPayload> {
        self.runtime_plan.as_ref()
    }

    pub fn runtime_execution_plan_output_sha256(&self) -> Option<String> {
        self.backend_policy.runtime_plan_output_sha256_hex()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_core::backend::device_caps::DeviceCaps;
    use st_tensor::TensorUtilBackend;

    #[test]
    fn compatibility_surface_shares_the_core_owned_policy_state() {
        let policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        let _guard = push_backend_policy(policy);

        assert_eq!(current_backend_policy(), Some(policy));
        assert_eq!(
            st_core::backend::execution::current_backend_policy(),
            Some(policy)
        );
        assert_eq!(current_tensor_util_backend(), TensorUtilBackend::Cpu);
    }

    #[test]
    fn trainer_context_keeps_uncommitted_planner_and_policy_aligned() {
        let caps = DeviceCaps::cpu();
        let config = ExecutionConfig::new(AcceleratorFallback::Forbid, 37);
        let context = TrainerExecutionContext::from_device_caps_with_config(caps, config);

        assert_eq!(context.planner().device_caps(), caps);
        assert_eq!(context.planner().execution_config(), config);
        assert_eq!(context.backend_policy().device_caps(), caps);
        assert_eq!(context.backend_policy().execution_config(), config);
        assert!(context.runtime_execution_plan().is_none());
        assert!(context.runtime_execution_plan_output_sha256().is_none());
    }
}
