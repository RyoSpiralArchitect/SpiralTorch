// SPDX-License-Identifier: AGPL-3.0-or-later

//! Kernel capability truth owned by the tensor implementation.
//!
//! Higher layers may commit or route these observations, but they must not
//! recreate backend/component support rules. Accelerator readiness combines an
//! exact workload preflight with a small operation-specific dispatch sentinel.

use serde::{Deserialize, Serialize};

/// Stable backend vocabulary used by tensor capability observations.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorExecutionBackend {
    Auto,
    Cpu,
    CpuSimd,
    Naive,
    Faer,
    Wgpu,
    Hip,
}

impl TensorExecutionBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::CpuSimd => "cpu_simd",
            Self::Naive => "naive",
            Self::Faer => "faer",
            Self::Wgpu => "wgpu",
            Self::Hip => "hip",
        }
    }

    pub const fn from_execution_id(value: &str) -> Option<Self> {
        match value.as_bytes() {
            b"auto" => Some(Self::Auto),
            b"cpu" => Some(Self::Cpu),
            b"cpu_simd" | b"simd" => Some(Self::CpuSimd),
            b"naive" => Some(Self::Naive),
            b"faer" => Some(Self::Faer),
            b"wgpu" | b"wgpu_dense" => Some(Self::Wgpu),
            b"hip" => Some(Self::Hip),
            _ => None,
        }
    }

    /// Whether this backend belongs to the component's stable execution vocabulary.
    pub const fn supports_component(self, component: TensorExecutionComponent) -> bool {
        match self {
            Self::Auto | Self::Wgpu => true,
            Self::Cpu => matches!(
                component,
                TensorExecutionComponent::LayerNorm
                    | TensorExecutionComponent::Attention
                    | TensorExecutionComponent::Softmax
                    | TensorExecutionComponent::TensorUtil
            ),
            Self::CpuSimd | Self::Naive | Self::Faer => matches!(
                component,
                TensorExecutionComponent::DenseMatmul | TensorExecutionComponent::PrepackedMatmul
            ),
            Self::Hip => matches!(component, TensorExecutionComponent::DenseMatmul),
        }
    }
}

/// Tensor operation families whose execution support can be observed.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorExecutionComponent {
    DenseMatmul,
    PrepackedMatmul,
    LayerNorm,
    Attention,
    Softmax,
    TensorUtil,
}

impl TensorExecutionComponent {
    pub const ALL: [Self; 6] = [
        Self::DenseMatmul,
        Self::PrepackedMatmul,
        Self::LayerNorm,
        Self::Attention,
        Self::Softmax,
        Self::TensorUtil,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DenseMatmul => "dense_matmul",
            Self::PrepackedMatmul => "prepacked_matmul",
            Self::LayerNorm => "layer_norm",
            Self::Attention => "attention",
            Self::Softmax => "softmax",
            Self::TensorUtil => "tensor_util",
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::DenseMatmul => 0,
            Self::PrepackedMatmul => 1,
            Self::LayerNorm => 2,
            Self::Attention => 3,
            Self::Softmax => 4,
            Self::TensorUtil => 5,
        }
    }
}

/// Tensor utility kernels represented by the current runtime contract.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TensorUtilOperation {
    Scale,
}

/// One concrete tensor workload whose implementation can be preflighted.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "component", rename_all = "snake_case", deny_unknown_fields)]
pub enum TensorExecutionWorkload {
    DenseMatmul {
        rows: u64,
        inner: u64,
        cols: u64,
    },
    PrepackedMatmul {
        rows: u64,
        inner: u64,
        cols: u64,
        #[serde(default)]
        bias: bool,
    },
    LayerNorm {
        rows: u64,
        cols: u64,
    },
    Attention {
        contexts: u64,
        sequence: u64,
        head_dim: u64,
        #[serde(default)]
        z_bias: bool,
        #[serde(default)]
        attn_bias: bool,
    },
    Softmax {
        rows: u64,
        cols: u64,
    },
    TensorUtil {
        operation: TensorUtilOperation,
        rows: u64,
        cols: u64,
    },
}

impl TensorExecutionWorkload {
    pub const fn component(self) -> TensorExecutionComponent {
        match self {
            Self::DenseMatmul { .. } => TensorExecutionComponent::DenseMatmul,
            Self::PrepackedMatmul { .. } => TensorExecutionComponent::PrepackedMatmul,
            Self::LayerNorm { .. } => TensorExecutionComponent::LayerNorm,
            Self::Attention { .. } => TensorExecutionComponent::Attention,
            Self::Softmax { .. } => TensorExecutionComponent::Softmax,
            Self::TensorUtil { .. } => TensorExecutionComponent::TensorUtil,
        }
    }

    pub fn output_values_saturating(self) -> usize {
        let volume = match self {
            Self::DenseMatmul { rows, cols, .. }
            | Self::PrepackedMatmul { rows, cols, .. }
            | Self::LayerNorm { rows, cols }
            | Self::Softmax { rows, cols }
            | Self::TensorUtil { rows, cols, .. } => rows.saturating_mul(cols),
            Self::Attention {
                contexts,
                sequence,
                head_dim,
                ..
            } => contexts.saturating_mul(sequence).saturating_mul(head_dim),
        };
        usize::try_from(volume).unwrap_or(usize::MAX)
    }

    pub fn has_empty_output(self) -> bool {
        self.output_values_saturating() == 0
    }
}

/// Capability state observed by the tensor implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorExecutionCapabilityStatus {
    Ready,
    Unavailable,
    NotBuilt,
    Unsupported,
}

/// Evidence required before a capability may be reported as ready.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorExecutionReadyProof {
    /// The host implementation is compiled and its complete shape contract holds.
    StaticHostContract,
    /// The exact accelerator workload passed preflight and its operation family
    /// completed a device dispatch plus readback sentinel.
    RuntimeDispatchSentinel,
}

/// Tensor-owned result for one backend/workload pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorExecutionCapability {
    pub backend: TensorExecutionBackend,
    pub workload: TensorExecutionWorkload,
    pub status: TensorExecutionCapabilityStatus,
    pub ready_proof: Option<TensorExecutionReadyProof>,
}

impl TensorExecutionCapability {
    fn ready(
        backend: TensorExecutionBackend,
        workload: TensorExecutionWorkload,
        proof: TensorExecutionReadyProof,
    ) -> Self {
        Self {
            backend,
            workload,
            status: TensorExecutionCapabilityStatus::Ready,
            ready_proof: Some(proof),
        }
    }

    fn unready(
        backend: TensorExecutionBackend,
        workload: TensorExecutionWorkload,
        status: TensorExecutionCapabilityStatus,
    ) -> Self {
        debug_assert_ne!(status, TensorExecutionCapabilityStatus::Ready);
        Self {
            backend,
            workload,
            status,
            ready_proof: None,
        }
    }
}

/// Observe the implementation that would execute one concrete tensor workload.
pub fn observe_tensor_execution_capability(
    backend: TensorExecutionBackend,
    workload: TensorExecutionWorkload,
) -> TensorExecutionCapability {
    if !workload_has_positive_dimensions(workload) {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unsupported,
        );
    }
    match backend {
        TensorExecutionBackend::Auto => TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unsupported,
        ),
        TensorExecutionBackend::Cpu => observe_host_capability(backend, workload, true),
        TensorExecutionBackend::CpuSimd | TensorExecutionBackend::Naive => {
            observe_host_capability(backend, workload, workload_is_matmul(workload))
        }
        TensorExecutionBackend::Faer => observe_faer_capability(backend, workload),
        TensorExecutionBackend::Wgpu => observe_wgpu_capability(backend, workload),
        TensorExecutionBackend::Hip => observe_hip_capability(backend, workload),
    }
}

fn workload_has_positive_dimensions(workload: TensorExecutionWorkload) -> bool {
    match workload {
        TensorExecutionWorkload::DenseMatmul { rows, inner, cols }
        | TensorExecutionWorkload::PrepackedMatmul {
            rows, inner, cols, ..
        } => rows > 0 && inner > 0 && cols > 0,
        TensorExecutionWorkload::LayerNorm { rows, cols }
        | TensorExecutionWorkload::Softmax { rows, cols }
        | TensorExecutionWorkload::TensorUtil { rows, cols, .. } => rows > 0 && cols > 0,
        TensorExecutionWorkload::Attention {
            contexts,
            sequence,
            head_dim,
            ..
        } => contexts > 0 && sequence > 0 && head_dim > 0,
    }
}

fn workload_is_matmul(workload: TensorExecutionWorkload) -> bool {
    matches!(
        workload,
        TensorExecutionWorkload::DenseMatmul { .. }
            | TensorExecutionWorkload::PrepackedMatmul { .. }
    )
}

fn observe_host_capability(
    backend: TensorExecutionBackend,
    workload: TensorExecutionWorkload,
    backend_supports_workload: bool,
) -> TensorExecutionCapability {
    if !backend_supports_workload {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unsupported,
        );
    }
    if !workload_fits_host(workload) {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unavailable,
        );
    }
    TensorExecutionCapability::ready(
        backend,
        workload,
        TensorExecutionReadyProof::StaticHostContract,
    )
}

fn observe_faer_capability(
    backend: TensorExecutionBackend,
    workload: TensorExecutionWorkload,
) -> TensorExecutionCapability {
    if !workload_is_matmul(workload) {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unsupported,
        );
    }
    if !crate::faer_dense::is_available() {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::NotBuilt,
        );
    }
    observe_host_capability(backend, workload, true)
}

fn workload_fits_host(workload: TensorExecutionWorkload) -> bool {
    match workload {
        TensorExecutionWorkload::DenseMatmul { rows, inner, cols } => {
            host_f32_buffer_fits(&[rows, inner])
                && host_f32_buffer_fits(&[inner, cols])
                && host_f32_buffer_fits(&[rows, cols])
        }
        TensorExecutionWorkload::PrepackedMatmul {
            rows,
            inner,
            cols,
            bias,
        } => {
            host_f32_buffer_fits(&[rows, inner])
                && host_f32_buffer_fits(&[inner, cols])
                && host_f32_buffer_fits(&[rows, cols])
                && (!bias || host_f32_buffer_fits(&[cols]))
        }
        TensorExecutionWorkload::LayerNorm { rows, cols }
        | TensorExecutionWorkload::Softmax { rows, cols }
        | TensorExecutionWorkload::TensorUtil { rows, cols, .. } => {
            host_f32_buffer_fits(&[rows, cols])
        }
        TensorExecutionWorkload::Attention {
            contexts,
            sequence,
            head_dim,
            z_bias,
            attn_bias,
        } => {
            host_f32_buffer_fits(&[contexts, sequence, head_dim])
                && (!z_bias || host_f32_buffer_fits(&[contexts, sequence]))
                && (!attn_bias || host_f32_buffer_fits(&[contexts, sequence, sequence]))
        }
    }
}

fn host_f32_buffer_fits(factors: &[u64]) -> bool {
    factors
        .iter()
        .try_fold(1_usize, |volume, factor| {
            let factor = usize::try_from(*factor).ok()?;
            volume.checked_mul(factor)
        })
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .is_some_and(|bytes| bytes <= isize::MAX as usize)
}

#[cfg(feature = "wgpu_dense")]
fn observe_wgpu_capability(
    backend: TensorExecutionBackend,
    workload: TensorExecutionWorkload,
) -> TensorExecutionCapability {
    if !crate::wgpu_dense::is_available() {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unavailable,
        );
    }
    if !wgpu_supports(workload) {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unsupported,
        );
    }
    if verify_wgpu_dispatch(workload).is_err() {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unavailable,
        );
    }
    TensorExecutionCapability::ready(
        backend,
        workload,
        TensorExecutionReadyProof::RuntimeDispatchSentinel,
    )
}

#[cfg(not(feature = "wgpu_dense"))]
fn observe_wgpu_capability(
    backend: TensorExecutionBackend,
    workload: TensorExecutionWorkload,
) -> TensorExecutionCapability {
    TensorExecutionCapability::unready(backend, workload, TensorExecutionCapabilityStatus::NotBuilt)
}

#[cfg(feature = "wgpu_dense")]
fn wgpu_supports(workload: TensorExecutionWorkload) -> bool {
    match workload {
        TensorExecutionWorkload::DenseMatmul { rows, inner, cols } => usize3(rows, inner, cols)
            .is_some_and(|(rows, inner, cols)| {
                crate::wgpu_dense::supports_matmul(rows, inner, cols)
            }),
        TensorExecutionWorkload::PrepackedMatmul {
            rows,
            inner,
            cols,
            bias,
        } => usize3(rows, inner, cols).is_some_and(|(rows, inner, cols)| {
            crate::wgpu_dense::supports_prepacked_matmul(rows, inner, cols, bias)
        }),
        TensorExecutionWorkload::LayerNorm { rows, cols } => dimensions2(rows, cols)
            .is_some_and(|(rows, cols)| crate::wgpu_dense::supports_layer_norm(rows, cols)),
        TensorExecutionWorkload::Attention {
            contexts,
            sequence,
            head_dim,
            z_bias,
            attn_bias,
        } => usize3(contexts, sequence, head_dim).is_some_and(|(contexts, sequence, head_dim)| {
            crate::wgpu_dense::supports_fused_attention_workload(
                contexts, sequence, head_dim, z_bias, attn_bias,
            )
        }),
        TensorExecutionWorkload::Softmax { rows, cols } => dimensions2(rows, cols)
            .is_some_and(|(rows, cols)| crate::wgpu_dense::supports_row_softmax(rows, cols)),
        TensorExecutionWorkload::TensorUtil {
            operation: TensorUtilOperation::Scale,
            rows,
            cols,
        } => dimensions2(rows, cols)
            .is_some_and(|(rows, cols)| crate::wgpu_dense::supports_tensor_util_scale(rows, cols)),
    }
}

#[cfg(feature = "wgpu_dense")]
fn verify_wgpu_dispatch(workload: TensorExecutionWorkload) -> Result<(), String> {
    match workload {
        TensorExecutionWorkload::DenseMatmul { .. } => verify_dispatch_output(
            "dense matmul",
            &crate::wgpu_dense::matmul(&[2.0], &[3.0], 1, 1, 1)?,
            &[6.0],
            1.0e-4,
        ),
        TensorExecutionWorkload::PrepackedMatmul { bias, .. } => {
            let rhs =
                crate::Tensor::from_vec(1, 1, vec![3.0]).map_err(|error| error.to_string())?;
            let packed = crate::PackedB::from_tensor(&rhs, crate::Tile::col_major())
                .map_err(|error| error.to_string())?;
            let output = if bias {
                crate::wgpu_dense::matmul_prepacked_bias(&[2.0], &packed, &[0.5], 1, 1, 1)?
            } else {
                crate::wgpu_dense::matmul_prepacked(&[2.0], &packed, 1, 1, 1)?
            };
            let expected = if bias { 6.5 } else { 6.0 };
            verify_dispatch_output("prepacked matmul", &output, &[expected], 1.0e-4)
        }
        TensorExecutionWorkload::LayerNorm { .. } => verify_dispatch_output(
            "layer norm",
            &crate::wgpu_dense::layer_norm_affine(
                &[1.0, 3.0],
                &[1.0, 1.0],
                &[0.0, 0.0],
                1,
                2,
                1.0e-5,
            )?,
            &[-0.999_995, 0.999_995],
            1.0e-3,
        ),
        TensorExecutionWorkload::Attention {
            z_bias, attn_bias, ..
        } => {
            let z_bias_values = [0.0_f32];
            let attn_bias_values = [0.0_f32];
            let output = crate::wgpu_dense::fused_attention(
                &[0.25, -0.5],
                &[0.75, 0.5],
                &[0.125, -0.25],
                1,
                1,
                2,
                std::f32::consts::FRAC_1_SQRT_2,
                z_bias.then_some(z_bias_values.as_slice()),
                attn_bias.then_some(attn_bias_values.as_slice()),
            )?;
            verify_dispatch_output("attention", &output, &[0.125, -0.25], 1.0e-4)
        }
        TensorExecutionWorkload::Softmax { .. } => verify_dispatch_output(
            "softmax",
            &crate::wgpu_dense::row_softmax(&[0.0, 1.0], 1, 2, crate::Layout::RowMajor)?,
            &[0.268_941_43, 0.731_058_6],
            1.0e-4,
        ),
        TensorExecutionWorkload::TensorUtil {
            operation: TensorUtilOperation::Scale,
            ..
        } => verify_dispatch_output(
            "tensor scale",
            &crate::wgpu_dense::scale(&[0.25, -0.5], 1, 2, 2.0)?,
            &[0.5, -1.0],
            1.0e-4,
        ),
    }
}

#[cfg(feature = "wgpu_dense")]
fn verify_dispatch_output(
    operation: &str,
    output: &[f32],
    expected: &[f32],
    tolerance: f32,
) -> Result<(), String> {
    if output.len() != expected.len() {
        return Err(format!(
            "{operation} dispatch sentinel returned {} values; expected {}",
            output.len(),
            expected.len()
        ));
    }
    for (index, (&actual, &expected)) in output.iter().zip(expected).enumerate() {
        if !actual.is_finite() || (actual - expected).abs() > tolerance {
            return Err(format!(
                "{operation} dispatch sentinel mismatch at {index}: got {actual}, expected {expected} within {tolerance}"
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "hip-real")]
fn observe_hip_capability(
    backend: TensorExecutionBackend,
    workload: TensorExecutionWorkload,
) -> TensorExecutionCapability {
    if !matches!(workload, TensorExecutionWorkload::DenseMatmul { .. }) {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unsupported,
        );
    }
    let TensorExecutionWorkload::DenseMatmul { rows, inner, cols } = workload else {
        unreachable!();
    };
    let Some((rows, inner, cols)) = usize3(rows, inner, cols) else {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unsupported,
        );
    };
    if !crate::backend::hip_dense::is_available() {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unavailable,
        );
    }
    if !crate::backend::hip_dense::supports_matmul(rows, inner, cols) {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unsupported,
        );
    }
    let mut output = [0.0_f32];
    if crate::backend::hip_dense::matmul_into(&[2.0], &[3.0], &mut output, 1, 1, 1).is_err()
        || !output[0].is_finite()
        || (output[0] - 6.0).abs() > 1.0e-4
    {
        return TensorExecutionCapability::unready(
            backend,
            workload,
            TensorExecutionCapabilityStatus::Unavailable,
        );
    }
    TensorExecutionCapability::ready(
        backend,
        workload,
        TensorExecutionReadyProof::RuntimeDispatchSentinel,
    )
}

#[cfg(not(feature = "hip-real"))]
fn observe_hip_capability(
    backend: TensorExecutionBackend,
    workload: TensorExecutionWorkload,
) -> TensorExecutionCapability {
    let status = if matches!(workload, TensorExecutionWorkload::DenseMatmul { .. }) {
        TensorExecutionCapabilityStatus::NotBuilt
    } else {
        TensorExecutionCapabilityStatus::Unsupported
    };
    TensorExecutionCapability::unready(backend, workload, status)
}

#[cfg(any(feature = "wgpu_dense", feature = "hip-real"))]
fn usize3(first: u64, second: u64, third: u64) -> Option<(usize, usize, usize)> {
    Some((
        first.try_into().ok()?,
        second.try_into().ok()?,
        third.try_into().ok()?,
    ))
}

#[cfg(feature = "wgpu_dense")]
fn dimensions2(first: u64, second: u64) -> Option<(usize, usize)> {
    let first: usize = first.try_into().ok()?;
    let second: usize = second.try_into().ok()?;
    first.checked_mul(second)?;
    Some((first, second))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_workload() -> TensorExecutionWorkload {
        TensorExecutionWorkload::DenseMatmul {
            rows: 2,
            inner: 3,
            cols: 4,
        }
    }

    #[test]
    fn backend_component_matrix_rejects_false_ready_claims() {
        let softmax = TensorExecutionWorkload::Softmax { rows: 2, cols: 4 };
        for backend in [
            TensorExecutionBackend::CpuSimd,
            TensorExecutionBackend::Naive,
            TensorExecutionBackend::Faer,
            TensorExecutionBackend::Hip,
        ] {
            let capability = observe_tensor_execution_capability(backend, softmax);
            assert_eq!(
                capability.status,
                TensorExecutionCapabilityStatus::Unsupported
            );
            assert_eq!(capability.ready_proof, None);
        }
    }

    #[test]
    fn static_host_ready_claims_carry_their_proof() {
        let capability =
            observe_tensor_execution_capability(TensorExecutionBackend::Cpu, dense_workload());
        assert_eq!(capability.status, TensorExecutionCapabilityStatus::Ready);
        assert_eq!(
            capability.ready_proof,
            Some(TensorExecutionReadyProof::StaticHostContract)
        );
    }

    #[test]
    fn malformed_workloads_cannot_report_ready_without_st_core_validation() {
        let workloads = [
            TensorExecutionWorkload::DenseMatmul {
                rows: 0,
                inner: 1,
                cols: 1,
            },
            TensorExecutionWorkload::PrepackedMatmul {
                rows: 1,
                inner: 0,
                cols: 1,
                bias: false,
            },
            TensorExecutionWorkload::LayerNorm { rows: 1, cols: 0 },
            TensorExecutionWorkload::Attention {
                contexts: 0,
                sequence: 1,
                head_dim: 1,
                z_bias: false,
                attn_bias: false,
            },
            TensorExecutionWorkload::Softmax { rows: 0, cols: 1 },
            TensorExecutionWorkload::TensorUtil {
                operation: TensorUtilOperation::Scale,
                rows: 1,
                cols: 0,
            },
        ];

        for workload in workloads {
            let capability =
                observe_tensor_execution_capability(TensorExecutionBackend::Cpu, workload);
            assert_eq!(
                capability.status,
                TensorExecutionCapabilityStatus::Unsupported
            );
            assert_eq!(capability.ready_proof, None);
        }
    }

    #[test]
    fn host_contract_rejects_buffers_outside_pointer_range() {
        let cols = (isize::MAX as u64 / std::mem::size_of::<f32>() as u64) + 1;
        let capability = observe_tensor_execution_capability(
            TensorExecutionBackend::Cpu,
            TensorExecutionWorkload::Softmax { rows: 1, cols },
        );
        assert_eq!(
            capability.status,
            TensorExecutionCapabilityStatus::Unavailable
        );
        assert_eq!(capability.ready_proof, None);
    }

    #[cfg(not(feature = "wgpu_dense"))]
    #[test]
    fn wgpu_without_the_feature_is_not_built() {
        let capability =
            observe_tensor_execution_capability(TensorExecutionBackend::Wgpu, dense_workload());
        assert_eq!(capability.status, TensorExecutionCapabilityStatus::NotBuilt);
        assert_eq!(capability.ready_proof, None);
    }

    #[cfg(feature = "wgpu_dense")]
    #[test]
    fn live_wgpu_ready_claims_require_operation_dispatch() {
        if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS")
            .ok()
            .as_deref()
            != Some("1")
        {
            eprintln!(
                "skipping live WGPU capability test; set SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1"
            );
            return;
        }

        let workloads = [
            dense_workload(),
            TensorExecutionWorkload::PrepackedMatmul {
                rows: 2,
                inner: 3,
                cols: 4,
                bias: true,
            },
            TensorExecutionWorkload::LayerNorm { rows: 2, cols: 4 },
            TensorExecutionWorkload::Attention {
                contexts: 1,
                sequence: 2,
                head_dim: 4,
                z_bias: true,
                attn_bias: true,
            },
            TensorExecutionWorkload::Softmax { rows: 2, cols: 4 },
            TensorExecutionWorkload::TensorUtil {
                operation: TensorUtilOperation::Scale,
                rows: 2,
                cols: 4,
            },
        ];

        for workload in workloads {
            let capability =
                observe_tensor_execution_capability(TensorExecutionBackend::Wgpu, workload);
            assert_eq!(capability.status, TensorExecutionCapabilityStatus::Ready);
            assert_eq!(
                capability.ready_proof,
                Some(TensorExecutionReadyProof::RuntimeDispatchSentinel)
            );
        }
    }
}
