// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Compatibility re-exports for the execution policy owned by `st-core`.

pub use st_core::backend::execution::{
    current_accelerator_fallback, current_attention_backend, current_backend_policy,
    current_layer_norm_backend, current_matmul_backend, current_prepacked_matmul_backend,
    current_softmax_backend, current_tensor_util_backend, current_tensor_util_backend_for_values,
    push_backend_policy, AcceleratorFallback, BackendPolicy, BackendPolicyGuard, ExecutionConfig,
};

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
}
