// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Lightweight execution policy shared between the trainer and tensor-heavy layers.

use serde_json::json;
use st_core::backend::device_caps::{BackendKind, DeviceCaps};
use st_tensor::{
    AttentionBackend, LayerNormBackend, MatmulBackend, SoftmaxBackend, TensorUtilBackend,
};
use std::cell::RefCell;

const DEFAULT_TENSOR_UTIL_WGPU_MIN_VALUES: usize = 1024;

thread_local! {
    static ACTIVE_BACKEND_POLICY: RefCell<Option<BackendPolicy>> = const { RefCell::new(None) };
}

/// Tensor backend policy derived from trainer-level device capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendPolicy {
    caps: DeviceCaps,
    matmul_backend: MatmulBackend,
    prepacked_matmul_backend: MatmulBackend,
    layer_norm_backend: LayerNormBackend,
    attention_backend: AttentionBackend,
    softmax_backend: SoftmaxBackend,
    tensor_util_backend: TensorUtilBackend,
}

impl BackendPolicy {
    /// Builds a policy from high-level device capabilities.
    pub fn from_device_caps(caps: DeviceCaps) -> Self {
        let matmul_backend = matmul_backend_for(caps.backend);
        let prepacked_matmul_backend = prepacked_matmul_backend_for(caps.backend);
        let layer_norm_backend = layer_norm_backend_for(caps.backend);
        let attention_backend = attention_backend_for(caps.backend);
        let softmax_backend = softmax_backend_for(caps.backend);
        let tensor_util_backend = tensor_util_backend_for(caps.backend);
        Self {
            caps,
            matmul_backend,
            prepacked_matmul_backend,
            layer_norm_backend,
            attention_backend,
            softmax_backend,
            tensor_util_backend,
        }
    }

    /// Creates a policy with explicit tensor backends. Useful for focused tests and experiments.
    pub fn explicit(
        caps: DeviceCaps,
        matmul_backend: MatmulBackend,
        prepacked_matmul_backend: MatmulBackend,
        layer_norm_backend: LayerNormBackend,
        attention_backend: AttentionBackend,
        softmax_backend: SoftmaxBackend,
    ) -> Self {
        Self {
            caps,
            matmul_backend,
            prepacked_matmul_backend,
            layer_norm_backend,
            attention_backend,
            softmax_backend,
            tensor_util_backend: tensor_util_backend_for(caps.backend),
        }
    }

    /// Returns the device capabilities that produced this policy.
    pub fn device_caps(self) -> DeviceCaps {
        self.caps
    }

    /// Backend used for dense `Tensor::matmul_with_backend` calls.
    pub fn matmul_backend(self) -> MatmulBackend {
        self.matmul_backend
    }

    /// Stable label for the device backend that produced this policy.
    pub fn device_backend_label(self) -> &'static str {
        self.caps.backend.as_str()
    }

    /// Stable label for the dense matmul policy backend.
    pub fn matmul_backend_label(self) -> &'static str {
        matmul_backend_label(self.matmul_backend)
    }

    /// Backend used for `Tensor::matmul_prepacked_with_backend` calls.
    pub fn prepacked_matmul_backend(self) -> MatmulBackend {
        self.prepacked_matmul_backend
    }

    /// Stable label for the prepacked matmul policy backend.
    pub fn prepacked_matmul_backend_label(self) -> &'static str {
        matmul_backend_label(self.prepacked_matmul_backend)
    }

    /// Backend used for tensor layer normalisation calls.
    pub fn layer_norm_backend(self) -> LayerNormBackend {
        self.layer_norm_backend
    }

    /// Stable label for the layer-normalisation policy backend.
    pub fn layer_norm_backend_label(self) -> &'static str {
        layer_norm_backend_label(self.layer_norm_backend)
    }

    /// Backend used for fused scaled dot-product attention calls.
    pub fn attention_backend(self) -> AttentionBackend {
        self.attention_backend
    }

    /// Stable label for the fused attention policy backend.
    pub fn attention_backend_label(self) -> &'static str {
        attention_backend_label(self.attention_backend)
    }

    /// Backend used for row-wise softmax calls.
    pub fn softmax_backend(self) -> SoftmaxBackend {
        self.softmax_backend
    }

    /// Stable label for the row-wise softmax policy backend.
    pub fn softmax_backend_label(self) -> &'static str {
        softmax_backend_label(self.softmax_backend)
    }

    /// Backend used for small tensor utility calls such as scale/reduce/projection.
    pub fn tensor_util_backend(self) -> TensorUtilBackend {
        self.tensor_util_backend
    }

    /// Stable label for the tensor utility policy backend.
    pub fn tensor_util_backend_label(self) -> &'static str {
        tensor_util_backend_label(self.tensor_util_backend)
    }
}

/// RAII guard that restores the previous thread-local backend policy.
#[derive(Debug)]
pub struct BackendPolicyGuard {
    previous: Option<BackendPolicy>,
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
    let previous = ACTIVE_BACKEND_POLICY.with(|slot| slot.replace(Some(policy)));
    BackendPolicyGuard { previous }
}

/// Returns the current thread-local backend policy, if any.
pub fn current_backend_policy() -> Option<BackendPolicy> {
    ACTIVE_BACKEND_POLICY.with(|slot| *slot.borrow())
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

/// Returns the current tensor utility backend, guarding small tensors from WGPU dispatch overhead.
pub fn current_tensor_util_backend_for_values(values: usize) -> TensorUtilBackend {
    let backend = current_tensor_util_backend();
    if matches!(backend, TensorUtilBackend::GpuWgpu) {
        let threshold = tensor_util_wgpu_min_values();
        let selected = if values < threshold {
            TensorUtilBackend::Cpu
        } else {
            backend
        };
        emit_tensor_util_route(values, threshold, backend, selected);
        return selected;
    }
    backend
}

fn emit_tensor_util_route(
    values: usize,
    threshold: usize,
    requested_backend: TensorUtilBackend,
    selected_backend: TensorUtilBackend,
) {
    let status = if matches!(selected_backend, TensorUtilBackend::GpuWgpu) {
        "wgpu"
    } else {
        "cpu_threshold"
    };
    st_tensor::emit_tensor_op_meta("tensor_util_route", || {
        json!({
            "requested_backend": tensor_util_backend_label(requested_backend),
            "selected_backend": tensor_util_backend_label(selected_backend),
            "status": status,
            "choice_source": "threshold_guard",
            "values": values,
            "threshold": threshold,
        })
    });
}

fn tensor_util_wgpu_min_values() -> usize {
    std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TENSOR_UTIL_WGPU_MIN_VALUES)
}

fn matmul_backend_for(kind: BackendKind) -> MatmulBackend {
    match kind {
        BackendKind::Wgpu => wgpu_matmul_backend(),
        BackendKind::Hip => hip_matmul_backend(),
        BackendKind::Cpu | BackendKind::Cuda | BackendKind::Mps => MatmulBackend::Auto,
    }
}

fn prepacked_matmul_backend_for(kind: BackendKind) -> MatmulBackend {
    match kind {
        BackendKind::Wgpu => wgpu_matmul_backend(),
        // HIP does not yet expose an explicit prepacked matmul path in st-tensor.
        BackendKind::Cpu | BackendKind::Cuda | BackendKind::Hip | BackendKind::Mps => {
            MatmulBackend::Auto
        }
    }
}

fn layer_norm_backend_for(kind: BackendKind) -> LayerNormBackend {
    match kind {
        BackendKind::Wgpu => wgpu_layer_norm_backend(),
        BackendKind::Cpu => LayerNormBackend::Cpu,
        BackendKind::Cuda | BackendKind::Hip | BackendKind::Mps => LayerNormBackend::Auto,
    }
}

fn attention_backend_for(kind: BackendKind) -> AttentionBackend {
    match kind {
        BackendKind::Wgpu => wgpu_attention_backend(),
        BackendKind::Cpu => AttentionBackend::Cpu,
        BackendKind::Cuda | BackendKind::Hip | BackendKind::Mps => AttentionBackend::Auto,
    }
}

fn softmax_backend_for(kind: BackendKind) -> SoftmaxBackend {
    match kind {
        BackendKind::Wgpu => wgpu_softmax_backend(),
        BackendKind::Cpu => SoftmaxBackend::Cpu,
        BackendKind::Cuda | BackendKind::Hip | BackendKind::Mps => SoftmaxBackend::Auto,
    }
}

fn tensor_util_backend_for(kind: BackendKind) -> TensorUtilBackend {
    match kind {
        BackendKind::Wgpu => TensorUtilBackend::GpuWgpu,
        BackendKind::Cpu => TensorUtilBackend::Cpu,
        BackendKind::Cuda | BackendKind::Hip | BackendKind::Mps => TensorUtilBackend::Auto,
    }
}

fn matmul_backend_label(backend: MatmulBackend) -> &'static str {
    match backend {
        MatmulBackend::Auto => "auto",
        MatmulBackend::CpuFaer => "faer",
        MatmulBackend::CpuSimd => "cpu_simd",
        MatmulBackend::CpuNaive => "naive",
        #[cfg(feature = "wgpu")]
        MatmulBackend::GpuWgpu => "wgpu",
        #[cfg(feature = "hip")]
        MatmulBackend::GpuHip => "hip",
        #[allow(unreachable_patterns)]
        _ => "gpu",
    }
}

fn layer_norm_backend_label(backend: LayerNormBackend) -> &'static str {
    match backend {
        LayerNormBackend::Auto => "auto",
        LayerNormBackend::Cpu => "cpu",
        LayerNormBackend::GpuWgpu => "wgpu",
    }
}

fn attention_backend_label(backend: AttentionBackend) -> &'static str {
    match backend {
        AttentionBackend::Auto => "auto",
        AttentionBackend::Cpu => "cpu",
        AttentionBackend::GpuWgpu => "wgpu",
    }
}

fn softmax_backend_label(backend: SoftmaxBackend) -> &'static str {
    match backend {
        SoftmaxBackend::Auto => "auto",
        SoftmaxBackend::Cpu => "cpu",
        #[cfg(feature = "wgpu")]
        SoftmaxBackend::GpuWgpu => "wgpu",
        #[allow(unreachable_patterns)]
        _ => "gpu",
    }
}

fn tensor_util_backend_label(backend: TensorUtilBackend) -> &'static str {
    match backend {
        TensorUtilBackend::Auto => "auto",
        TensorUtilBackend::Cpu => "cpu",
        TensorUtilBackend::GpuWgpu => "wgpu",
    }
}

fn wgpu_matmul_backend() -> MatmulBackend {
    #[cfg(feature = "wgpu")]
    {
        MatmulBackend::GpuWgpu
    }
    #[cfg(not(feature = "wgpu"))]
    {
        MatmulBackend::Auto
    }
}

fn hip_matmul_backend() -> MatmulBackend {
    #[cfg(feature = "hip")]
    {
        MatmulBackend::GpuHip
    }
    #[cfg(not(feature = "hip"))]
    {
        MatmulBackend::Auto
    }
}

fn wgpu_layer_norm_backend() -> LayerNormBackend {
    #[cfg(feature = "wgpu")]
    {
        LayerNormBackend::GpuWgpu
    }
    #[cfg(not(feature = "wgpu"))]
    {
        LayerNormBackend::Auto
    }
}

fn wgpu_attention_backend() -> AttentionBackend {
    #[cfg(feature = "wgpu")]
    {
        AttentionBackend::GpuWgpu
    }
    #[cfg(not(feature = "wgpu"))]
    {
        AttentionBackend::Auto
    }
}

fn wgpu_softmax_backend() -> SoftmaxBackend {
    #[cfg(feature = "wgpu")]
    {
        SoftmaxBackend::GpuWgpu
    }
    #[cfg(not(feature = "wgpu"))]
    {
        SoftmaxBackend::Auto
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex, OnceLock};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn cpu_policy_keeps_tensor_cpu_only_for_accelerated_ops() {
        let policy = BackendPolicy::from_device_caps(DeviceCaps::cpu());
        assert_eq!(policy.matmul_backend(), MatmulBackend::Auto);
        assert_eq!(policy.prepacked_matmul_backend(), MatmulBackend::Auto);
        assert_eq!(policy.layer_norm_backend(), LayerNormBackend::Cpu);
        assert_eq!(policy.attention_backend(), AttentionBackend::Cpu);
        assert_eq!(policy.softmax_backend(), SoftmaxBackend::Cpu);
        assert_eq!(policy.tensor_util_backend(), TensorUtilBackend::Cpu);
        assert_eq!(policy.device_backend_label(), "cpu");
        assert_eq!(policy.matmul_backend_label(), "auto");
        assert_eq!(policy.prepacked_matmul_backend_label(), "auto");
        assert_eq!(policy.layer_norm_backend_label(), "cpu");
        assert_eq!(policy.attention_backend_label(), "cpu");
        assert_eq!(policy.softmax_backend_label(), "cpu");
        assert_eq!(policy.tensor_util_backend_label(), "cpu");
    }

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
    fn tensor_util_threshold_route_emits_policy_meta_without_kernel_backend() {
        let _lock = observer_lock();
        let previous_threshold = std::env::var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES").ok();
        std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", "1024");

        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous_observer =
            st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
                captured
                    .lock()
                    .unwrap()
                    .push((event.op_name, event.data.clone()));
            })));

        let policy = BackendPolicy::from_device_caps(DeviceCaps::wgpu(32, true, 256));
        let guard = push_backend_policy(policy);
        let selected = current_tensor_util_backend_for_values(8);
        drop(guard);

        st_tensor::set_tensor_op_meta_observer(previous_observer);
        match previous_threshold {
            Some(value) => std::env::set_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES", value),
            None => std::env::remove_var("SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES"),
        }

        assert_eq!(selected, TensorUtilBackend::Cpu);
        let events = events.lock().unwrap();
        let (_, data) = events
            .iter()
            .find(|(op_name, _)| *op_name == "tensor_util_route")
            .expect("tensor util route metadata");
        assert_eq!(data["requested_backend"], "wgpu");
        assert_eq!(data["selected_backend"], "cpu");
        assert_eq!(data["status"], "cpu_threshold");
        assert_eq!(data["values"], 8);
        assert_eq!(data["threshold"], 1024);
        assert!(data.get("backend").is_none());
    }
}
