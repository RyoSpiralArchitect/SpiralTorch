//! Typed tensor execution plans shared by Rust trainers and language bindings.

use super::device_caps::{BackendKind, DeviceCaps};
pub use spiral_config::execution::{AcceleratorFallback, ExecutionConfig};
use st_tensor::{
    AttentionBackend, LayerNormBackend, MatmulBackend, SoftmaxBackend, TensorUtilBackend,
};

/// Tensor backend policy derived from device capabilities and captured runtime configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendPolicy {
    caps: DeviceCaps,
    config: ExecutionConfig,
    matmul_backend: MatmulBackend,
    prepacked_matmul_backend: MatmulBackend,
    layer_norm_backend: LayerNormBackend,
    attention_backend: AttentionBackend,
    softmax_backend: SoftmaxBackend,
    tensor_util_backend: TensorUtilBackend,
}

impl BackendPolicy {
    /// Builds a policy from device capabilities and captures process configuration once.
    pub fn from_device_caps(caps: DeviceCaps) -> Self {
        Self::from_device_caps_with_config(caps, ExecutionConfig::from_env())
    }

    /// Builds a deterministic policy from explicit device capabilities and configuration.
    pub fn from_device_caps_with_config(caps: DeviceCaps, config: ExecutionConfig) -> Self {
        Self {
            caps,
            config,
            matmul_backend: matmul_backend_for(caps.backend),
            prepacked_matmul_backend: prepacked_matmul_backend_for(caps.backend),
            layer_norm_backend: layer_norm_backend_for(caps.backend),
            attention_backend: attention_backend_for(caps.backend),
            softmax_backend: softmax_backend_for(caps.backend),
            tensor_util_backend: tensor_util_backend_for(caps.backend),
        }
    }

    /// Creates a policy with explicit tensor backends for focused tests and experiments.
    pub fn explicit(
        caps: DeviceCaps,
        matmul_backend: MatmulBackend,
        prepacked_matmul_backend: MatmulBackend,
        layer_norm_backend: LayerNormBackend,
        attention_backend: AttentionBackend,
        softmax_backend: SoftmaxBackend,
    ) -> Self {
        Self::explicit_with_config(
            caps,
            ExecutionConfig::from_env(),
            matmul_backend,
            prepacked_matmul_backend,
            layer_norm_backend,
            attention_backend,
            softmax_backend,
        )
    }

    /// Creates an explicit policy with a deterministic execution configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn explicit_with_config(
        caps: DeviceCaps,
        config: ExecutionConfig,
        matmul_backend: MatmulBackend,
        prepacked_matmul_backend: MatmulBackend,
        layer_norm_backend: LayerNormBackend,
        attention_backend: AttentionBackend,
        softmax_backend: SoftmaxBackend,
    ) -> Self {
        Self {
            caps,
            config,
            matmul_backend,
            prepacked_matmul_backend,
            layer_norm_backend,
            attention_backend,
            softmax_backend,
            tensor_util_backend: tensor_util_backend_for(caps.backend),
        }
    }

    pub const fn device_caps(self) -> DeviceCaps {
        self.caps
    }

    pub const fn execution_config(self) -> ExecutionConfig {
        self.config
    }

    pub const fn matmul_backend(self) -> MatmulBackend {
        self.matmul_backend
    }

    pub const fn prepacked_matmul_backend(self) -> MatmulBackend {
        self.prepacked_matmul_backend
    }

    pub const fn layer_norm_backend(self) -> LayerNormBackend {
        self.layer_norm_backend
    }

    pub const fn attention_backend(self) -> AttentionBackend {
        self.attention_backend
    }

    pub const fn softmax_backend(self) -> SoftmaxBackend {
        self.softmax_backend
    }

    pub const fn tensor_util_backend(self) -> TensorUtilBackend {
        self.tensor_util_backend
    }

    pub const fn device_backend_label(self) -> &'static str {
        self.caps.backend.as_str()
    }

    pub fn matmul_backend_label(self) -> &'static str {
        matmul_backend_label(self.matmul_backend)
    }

    pub fn prepacked_matmul_backend_label(self) -> &'static str {
        matmul_backend_label(self.prepacked_matmul_backend)
    }

    pub fn layer_norm_backend_label(self) -> &'static str {
        layer_norm_backend_label(self.layer_norm_backend)
    }

    pub fn attention_backend_label(self) -> &'static str {
        attention_backend_label(self.attention_backend)
    }

    pub fn softmax_backend_label(self) -> &'static str {
        softmax_backend_label(self.softmax_backend)
    }

    pub fn tensor_util_backend_label(self) -> &'static str {
        tensor_util_backend_label(self.tensor_util_backend)
    }

    /// Resolves the utility-kernel route without consulting mutable global state.
    pub fn tensor_util_route(self, values: usize) -> TensorUtilRoute {
        let requested_backend = self.tensor_util_backend;
        let threshold = self.config.tensor_util_wgpu_min_values;
        let (selected_backend, status) =
            if matches!(requested_backend, TensorUtilBackend::GpuWgpu) && values < threshold {
                (TensorUtilBackend::Cpu, TensorUtilRouteStatus::CpuThreshold)
            } else if matches!(requested_backend, TensorUtilBackend::GpuWgpu) {
                (requested_backend, TensorUtilRouteStatus::Wgpu)
            } else {
                (requested_backend, TensorUtilRouteStatus::Direct)
            };

        TensorUtilRoute {
            requested_backend,
            selected_backend,
            values,
            threshold,
            status,
        }
    }
}

/// Result of applying the typed utility-kernel threshold to one tensor operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorUtilRoute {
    pub requested_backend: TensorUtilBackend,
    pub selected_backend: TensorUtilBackend,
    pub values: usize,
    pub threshold: usize,
    pub status: TensorUtilRouteStatus,
}

impl TensorUtilRoute {
    pub fn requested_backend_label(self) -> &'static str {
        tensor_util_backend_label(self.requested_backend)
    }

    pub fn selected_backend_label(self) -> &'static str {
        tensor_util_backend_label(self.selected_backend)
    }

    pub const fn records_threshold_decision(self) -> bool {
        matches!(
            self.status,
            TensorUtilRouteStatus::Wgpu | TensorUtilRouteStatus::CpuThreshold
        )
    }
}

/// Stable route outcome vocabulary shared by telemetry and language bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorUtilRouteStatus {
    Direct,
    Wgpu,
    CpuThreshold,
}

impl TensorUtilRouteStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Wgpu => "wgpu",
            Self::CpuThreshold => "cpu_threshold",
        }
    }
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
    use spiral_config::execution::{AcceleratorFallback, ExecutionConfig};

    #[test]
    fn cpu_policy_keeps_accelerated_ops_on_cpu_or_auto() {
        let policy = BackendPolicy::from_device_caps_with_config(
            DeviceCaps::cpu(),
            ExecutionConfig::default(),
        );
        assert_eq!(policy.matmul_backend(), MatmulBackend::Auto);
        assert_eq!(policy.prepacked_matmul_backend(), MatmulBackend::Auto);
        assert_eq!(policy.layer_norm_backend(), LayerNormBackend::Cpu);
        assert_eq!(policy.attention_backend(), AttentionBackend::Cpu);
        assert_eq!(policy.softmax_backend(), SoftmaxBackend::Cpu);
        assert_eq!(policy.tensor_util_backend(), TensorUtilBackend::Cpu);
        assert_eq!(policy.device_backend_label(), "cpu");
        assert_eq!(policy.tensor_util_backend_label(), "cpu");
    }

    #[test]
    fn tensor_utility_threshold_is_part_of_the_captured_plan() {
        let config = ExecutionConfig::new(AcceleratorFallback::Forbid, 1024);
        let policy =
            BackendPolicy::from_device_caps_with_config(DeviceCaps::wgpu(32, true, 256), config);

        let small = policy.tensor_util_route(8);
        assert_eq!(small.requested_backend, TensorUtilBackend::GpuWgpu);
        assert_eq!(small.selected_backend, TensorUtilBackend::Cpu);
        assert_eq!(small.status, TensorUtilRouteStatus::CpuThreshold);
        assert_eq!(small.threshold, 1024);

        let large = policy.tensor_util_route(1024);
        assert_eq!(large.selected_backend, TensorUtilBackend::GpuWgpu);
        assert_eq!(large.status, TensorUtilRouteStatus::Wgpu);
        assert_eq!(
            policy.execution_config().accelerator_fallback,
            AcceleratorFallback::Forbid
        );
    }
}
