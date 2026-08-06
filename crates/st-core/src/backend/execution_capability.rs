//! Workload-specific runtime capability evidence for tensor execution plans.
//!
//! Capability observation is intentionally separate from plan evaluation.
//! Native Rust observes mutable local runtime state once, then the execution
//! plan commits the typed evidence so validation and cross-language replay do
//! not query the device again.

use super::execution_plan::{RuntimeExecutionComponent, RuntimeTensorBackend};
use serde::{Deserialize, Serialize};

/// Tensor-utility kernels that can currently be preflighted by an execution plan.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeTensorUtilOperation {
    Scale,
}

/// One concrete workload whose native implementation can be observed.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "component", rename_all = "snake_case", deny_unknown_fields)]
pub enum RuntimeComponentWorkload {
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
    /// Row-major row softmax. Other tensor layouts require separate evidence.
    Softmax {
        rows: u64,
        cols: u64,
    },
    TensorUtil {
        operation: RuntimeTensorUtilOperation,
        rows: u64,
        cols: u64,
    },
}

impl RuntimeComponentWorkload {
    pub const fn component(&self) -> RuntimeExecutionComponent {
        match self {
            Self::DenseMatmul { .. } => RuntimeExecutionComponent::DenseMatmul,
            Self::PrepackedMatmul { .. } => RuntimeExecutionComponent::PrepackedMatmul,
            Self::LayerNorm { .. } => RuntimeExecutionComponent::LayerNorm,
            Self::Attention { .. } => RuntimeExecutionComponent::Attention,
            Self::Softmax { .. } => RuntimeExecutionComponent::Softmax,
            Self::TensorUtil { .. } => RuntimeExecutionComponent::TensorUtil,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        match self {
            Self::DenseMatmul { rows, inner, cols }
            | Self::PrepackedMatmul {
                rows, inner, cols, ..
            } => validate_dimensions(
                self.component(),
                &[("rows", *rows), ("inner", *inner), ("cols", *cols)],
                &[
                    ("lhs", &[*rows, *inner]),
                    ("rhs", &[*inner, *cols]),
                    ("output", &[*rows, *cols]),
                ],
            ),
            Self::LayerNorm { rows, cols }
            | Self::Softmax { rows, cols }
            | Self::TensorUtil { rows, cols, .. } => validate_dimensions(
                self.component(),
                &[("rows", *rows), ("cols", *cols)],
                &[("values", &[*rows, *cols])],
            ),
            Self::Attention {
                contexts,
                sequence,
                head_dim,
                z_bias,
                attn_bias,
            } => {
                validate_dimensions(
                    self.component(),
                    &[
                        ("contexts", *contexts),
                        ("sequence", *sequence),
                        ("head_dim", *head_dim),
                    ],
                    &[("values", &[*contexts, *sequence, *head_dim])],
                )?;
                if *z_bias {
                    validate_volume(self.component(), "z_bias", &[*contexts, *sequence])?;
                }
                if *attn_bias {
                    validate_volume(
                        self.component(),
                        "attn_bias",
                        &[*contexts, *sequence, *sequence],
                    )?;
                }
                Ok(())
            }
        }
    }
}

fn validate_dimensions(
    component: RuntimeExecutionComponent,
    dimensions: &[(&str, u64)],
    volumes: &[(&str, &[u64])],
) -> Result<(), String> {
    for (name, value) in dimensions {
        if *value == 0 {
            return Err(format!(
                "{} workload dimension '{name}' must be positive",
                component.as_str()
            ));
        }
    }
    for (name, factors) in volumes {
        validate_volume(component, name, factors)?;
    }
    Ok(())
}

fn validate_volume(
    component: RuntimeExecutionComponent,
    name: &str,
    factors: &[u64],
) -> Result<(), String> {
    if factors
        .iter()
        .try_fold(1_u64, |volume, factor| volume.checked_mul(*factor))
        .is_none()
    {
        return Err(format!(
            "{} workload volume '{name}' exceeds u64 range",
            component.as_str()
        ));
    }
    Ok(())
}

/// Result captured by the Rust local capability observer.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeComponentCapabilityStatus {
    Ready,
    Unavailable,
    NotBuilt,
    Unsupported,
}

impl RuntimeComponentCapabilityStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Unavailable => "unavailable",
            Self::NotBuilt => "not_built",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Committed observation for one component workload and selected backend.
///
/// This is deterministic replay evidence, not cryptographic hardware
/// attestation. Local callers should obtain it from the Rust observer and
/// retain the enclosing request commitment.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeComponentCapabilityEvidence {
    pub workload: RuntimeComponentWorkload,
    pub backend: RuntimeTensorBackend,
    pub status: RuntimeComponentCapabilityStatus,
}

impl RuntimeComponentCapabilityEvidence {
    pub const fn component(&self) -> RuntimeExecutionComponent {
        self.workload.component()
    }

    pub(crate) fn validate(&self) -> Result<(), String> {
        self.workload.validate()
    }
}

/// Capability state projected onto a component route.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeComponentCapabilityState {
    /// The compiled host implementation does not require a mutable runtime probe.
    Static,
    Ready,
    Unobserved,
    Unavailable,
    NotBuilt,
    Unsupported,
    NotApplicable,
}

impl RuntimeComponentCapabilityState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Ready => "ready",
            Self::Unobserved => "unobserved",
            Self::Unavailable => "unavailable",
            Self::NotBuilt => "not_built",
            Self::Unsupported => "unsupported",
            Self::NotApplicable => "not_applicable",
        }
    }

    pub const fn is_ready(self) -> bool {
        matches!(self, Self::Static | Self::Ready)
    }
}

impl From<RuntimeComponentCapabilityStatus> for RuntimeComponentCapabilityState {
    fn from(status: RuntimeComponentCapabilityStatus) -> Self {
        match status {
            RuntimeComponentCapabilityStatus::Ready => Self::Ready,
            RuntimeComponentCapabilityStatus::Unavailable => Self::Unavailable,
            RuntimeComponentCapabilityStatus::NotBuilt => Self::NotBuilt,
            RuntimeComponentCapabilityStatus::Unsupported => Self::Unsupported,
        }
    }
}

pub(crate) fn observe_component_capability(
    backend: RuntimeTensorBackend,
    workload: RuntimeComponentWorkload,
) -> RuntimeComponentCapabilityEvidence {
    let status = match backend {
        RuntimeTensorBackend::Cpu | RuntimeTensorBackend::CpuSimd | RuntimeTensorBackend::Naive => {
            observe_builtin_host_capability(&workload)
        }
        RuntimeTensorBackend::Faer => observe_faer_capability(&workload),
        RuntimeTensorBackend::Wgpu => observe_wgpu_capability(&workload),
        RuntimeTensorBackend::Hip => observe_hip_capability(&workload),
        RuntimeTensorBackend::Auto => RuntimeComponentCapabilityStatus::Unsupported,
    };
    RuntimeComponentCapabilityEvidence {
        workload,
        backend,
        status,
    }
}

fn observe_builtin_host_capability(
    workload: &RuntimeComponentWorkload,
) -> RuntimeComponentCapabilityStatus {
    if workload_fits_target(workload) {
        RuntimeComponentCapabilityStatus::Ready
    } else {
        RuntimeComponentCapabilityStatus::Unavailable
    }
}

fn observe_faer_capability(
    workload: &RuntimeComponentWorkload,
) -> RuntimeComponentCapabilityStatus {
    if !st_tensor::faer_dense::is_available() {
        return RuntimeComponentCapabilityStatus::NotBuilt;
    }
    match workload {
        RuntimeComponentWorkload::DenseMatmul { .. }
        | RuntimeComponentWorkload::PrepackedMatmul { .. }
            if workload_fits_target(workload) =>
        {
            RuntimeComponentCapabilityStatus::Ready
        }
        RuntimeComponentWorkload::DenseMatmul { .. }
        | RuntimeComponentWorkload::PrepackedMatmul { .. } => {
            RuntimeComponentCapabilityStatus::Unavailable
        }
        _ => RuntimeComponentCapabilityStatus::Unsupported,
    }
}

fn workload_fits_target(workload: &RuntimeComponentWorkload) -> bool {
    match workload {
        RuntimeComponentWorkload::DenseMatmul { rows, inner, cols } => {
            host_f32_buffer_fits(&[*rows, *inner])
                && host_f32_buffer_fits(&[*inner, *cols])
                && host_f32_buffer_fits(&[*rows, *cols])
        }
        RuntimeComponentWorkload::PrepackedMatmul {
            rows,
            inner,
            cols,
            bias,
        } => {
            host_f32_buffer_fits(&[*rows, *inner])
                && host_f32_buffer_fits(&[*inner, *cols])
                && host_f32_buffer_fits(&[*rows, *cols])
                && (!*bias || host_f32_buffer_fits(&[*cols]))
        }
        RuntimeComponentWorkload::LayerNorm { rows, cols }
        | RuntimeComponentWorkload::Softmax { rows, cols }
        | RuntimeComponentWorkload::TensorUtil { rows, cols, .. } => {
            host_f32_buffer_fits(&[*rows, *cols])
        }
        RuntimeComponentWorkload::Attention {
            contexts,
            sequence,
            head_dim,
            z_bias,
            attn_bias,
        } => {
            host_f32_buffer_fits(&[*contexts, *sequence, *head_dim])
                && (!*z_bias || host_f32_buffer_fits(&[*contexts, *sequence]))
                && (!*attn_bias || host_f32_buffer_fits(&[*contexts, *sequence, *sequence]))
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

#[cfg(feature = "wgpu")]
fn observe_wgpu_capability(
    workload: &RuntimeComponentWorkload,
) -> RuntimeComponentCapabilityStatus {
    use RuntimeComponentCapabilityStatus::{Ready, Unavailable};

    let supported = match workload {
        RuntimeComponentWorkload::DenseMatmul { rows, inner, cols } => usize3(*rows, *inner, *cols)
            .is_some_and(|(rows, inner, cols)| {
                st_tensor::wgpu_dense::supports_matmul(rows, inner, cols)
            }),
        RuntimeComponentWorkload::PrepackedMatmul {
            rows,
            inner,
            cols,
            bias,
        } => usize3(*rows, *inner, *cols).is_some_and(|(rows, inner, cols)| {
            st_tensor::wgpu_dense::supports_prepacked_matmul(rows, inner, cols, *bias)
        }),
        RuntimeComponentWorkload::LayerNorm { rows, cols } => dimensions2(*rows, *cols)
            .is_some_and(|(rows, cols)| st_tensor::wgpu_dense::supports_layer_norm(rows, cols)),
        RuntimeComponentWorkload::Attention {
            contexts,
            sequence,
            head_dim,
            z_bias,
            attn_bias,
        } => dimensions3(*contexts, *sequence, *head_dim).is_some_and(
            |(contexts, sequence, head_dim)| {
                st_tensor::wgpu_dense::supports_fused_attention_workload(
                    contexts, sequence, head_dim, *z_bias, *attn_bias,
                )
            },
        ),
        RuntimeComponentWorkload::Softmax { rows, cols } => dimensions2(*rows, *cols)
            .is_some_and(|(rows, cols)| st_tensor::wgpu_dense::supports_row_softmax(rows, cols)),
        RuntimeComponentWorkload::TensorUtil {
            operation,
            rows,
            cols,
        } => dimensions2(*rows, *cols).is_some_and(|(rows, cols)| match operation {
            RuntimeTensorUtilOperation::Scale => {
                st_tensor::wgpu_dense::supports_tensor_util_scale(rows, cols)
            }
        }),
    };
    if supported {
        Ready
    } else {
        Unavailable
    }
}

#[cfg(not(feature = "wgpu"))]
fn observe_wgpu_capability(
    _workload: &RuntimeComponentWorkload,
) -> RuntimeComponentCapabilityStatus {
    RuntimeComponentCapabilityStatus::NotBuilt
}

#[cfg(feature = "hip-real")]
fn observe_hip_capability(workload: &RuntimeComponentWorkload) -> RuntimeComponentCapabilityStatus {
    match workload {
        RuntimeComponentWorkload::DenseMatmul { rows, inner, cols }
            if usize3(*rows, *inner, *cols).is_some_and(|(rows, inner, cols)| {
                st_tensor::backend::hip_dense::supports_matmul(rows, inner, cols)
            }) =>
        {
            RuntimeComponentCapabilityStatus::Ready
        }
        RuntimeComponentWorkload::DenseMatmul { .. } => {
            RuntimeComponentCapabilityStatus::Unavailable
        }
        _ => RuntimeComponentCapabilityStatus::Unsupported,
    }
}

#[cfg(not(feature = "hip-real"))]
fn observe_hip_capability(
    _workload: &RuntimeComponentWorkload,
) -> RuntimeComponentCapabilityStatus {
    RuntimeComponentCapabilityStatus::NotBuilt
}

#[cfg(feature = "wgpu")]
fn dimensions2(first: u64, second: u64) -> Option<(usize, usize)> {
    let first: usize = first.try_into().ok()?;
    let second: usize = second.try_into().ok()?;
    first.checked_mul(second)?;
    Some((first, second))
}

#[cfg(feature = "wgpu")]
fn dimensions3(first: u64, second: u64, third: u64) -> Option<(usize, usize, usize)> {
    let dimensions = usize3(first, second, third)?;
    dimensions
        .0
        .checked_mul(dimensions.1)?
        .checked_mul(dimensions.2)?;
    Some(dimensions)
}

#[cfg(any(feature = "wgpu", feature = "hip-real"))]
fn usize3(first: u64, second: u64, third: u64) -> Option<(usize, usize, usize)> {
    let first: usize = first.try_into().ok()?;
    let second: usize = second.try_into().ok()?;
    let third: usize = third.try_into().ok()?;
    Some((first, second, third))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workload_validation_rejects_zero_and_overflow() {
        let zero = RuntimeComponentWorkload::Softmax { rows: 0, cols: 8 };
        assert!(zero.validate().unwrap_err().contains("must be positive"));

        let overflow = RuntimeComponentWorkload::Attention {
            contexts: u64::MAX,
            sequence: 2,
            head_dim: 2,
            z_bias: false,
            attn_bias: false,
        };
        assert!(overflow
            .validate()
            .unwrap_err()
            .contains("exceeds u64 range"));

        let bias_overflow = RuntimeComponentWorkload::Attention {
            contexts: 1,
            sequence: u64::MAX,
            head_dim: 1,
            z_bias: false,
            attn_bias: true,
        };
        assert!(bias_overflow.validate().unwrap_err().contains("attn_bias"));
    }

    #[test]
    fn host_capability_observation_is_ready_evidence() {
        let workload = RuntimeComponentWorkload::DenseMatmul {
            rows: 2,
            inner: 3,
            cols: 4,
        };
        let evidence = observe_component_capability(RuntimeTensorBackend::Faer, workload.clone());
        assert_eq!(evidence.workload, workload);
        assert_eq!(evidence.backend, RuntimeTensorBackend::Faer);
        assert_eq!(evidence.status, RuntimeComponentCapabilityStatus::Ready);
    }

    #[test]
    fn host_capability_rejects_buffers_outside_the_pointer_range() {
        let cols = (isize::MAX as u64 / std::mem::size_of::<f32>() as u64) + 1;
        let workload = RuntimeComponentWorkload::Softmax { rows: 1, cols };

        assert!(workload.validate().is_ok());
        assert!(!workload_fits_target(&workload));
        assert_eq!(
            observe_component_capability(RuntimeTensorBackend::Cpu, workload).status,
            RuntimeComponentCapabilityStatus::Unavailable
        );
    }
}
