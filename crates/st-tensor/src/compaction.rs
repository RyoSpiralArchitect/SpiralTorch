// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tensor-facing row compaction over one shared kernel contract.
//!
//! CPU, WGPU, and HIP remain explicit execution choices here. Runtime policy
//! belongs to the Rust orchestration layer rather than this operation facade.

use std::error::Error;
use std::fmt;

pub use st_kernel_contracts::compaction::{
    compact_rows_reference_f32, tiles_per_row, CompactionError, CompactionLayout,
    CompactionOutputF32, CompactionShape, COMPACTION_TILE,
};

#[non_exhaustive]
#[derive(Debug)]
pub enum TensorCompactionError {
    Contract(CompactionError),
    #[cfg(feature = "wgpu_dense")]
    Wgpu(st_backend_wgpu::compaction::CompactionDispatchError),
    Backend(String),
}

impl fmt::Display for TensorCompactionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(formatter, "compaction contract: {error}"),
            #[cfg(feature = "wgpu_dense")]
            Self::Wgpu(error) => write!(formatter, "WGPU compaction: {error}"),
            Self::Backend(error) => write!(formatter, "compaction backend: {error}"),
        }
    }
}

impl Error for TensorCompactionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            #[cfg(feature = "wgpu_dense")]
            Self::Wgpu(error) => Some(error),
            Self::Backend(_) => None,
        }
    }
}

#[cfg(feature = "wgpu_dense")]
impl From<st_backend_wgpu::compaction::CompactionDispatchError> for TensorCompactionError {
    fn from(error: st_backend_wgpu::compaction::CompactionDispatchError) -> Self {
        Self::Wgpu(error)
    }
}

impl From<CompactionError> for TensorCompactionError {
    fn from(error: CompactionError) -> Self {
        Self::Contract(error)
    }
}

/// Runs the canonical CPU oracle through the tensor operation surface.
pub fn compact_rows_cpu_f32(
    values: &[f32],
    indices: &[i32],
    shape: CompactionShape,
) -> Result<CompactionOutputF32, TensorCompactionError> {
    compact_rows_reference_f32(values, indices, shape).map_err(TensorCompactionError::from)
}

/// Runs the owning WGPU executor against the same compaction contract.
#[cfg(feature = "wgpu_dense")]
pub fn compact_rows_wgpu_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    indices: &[i32],
    shape: CompactionShape,
) -> Result<CompactionOutputF32, TensorCompactionError> {
    st_backend_wgpu::compaction::compact_rows_f32(device, queue, values, indices, shape)
        .map_err(TensorCompactionError::from)
}

/// Runs the owning HIP executor against the same compaction contract.
#[cfg(feature = "hip-real")]
pub fn compact_rows_hip_f32(
    values: &[f32],
    indices: &[i32],
    shape: CompactionShape,
) -> Result<CompactionOutputF32, TensorCompactionError> {
    crate::backend::hip_dense::ensure_runtime().map_err(TensorCompactionError::Backend)?;
    st_backend_hip::compaction::compact_rows_f32(values, indices, shape)
        .map_err(|error| TensorCompactionError::Backend(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_entrypoint_preserves_the_shared_contract_type() {
        let output = compact_rows_cpu_f32(
            &[0.2, 0.7, 0.5, f32::NAN],
            &[1, 2, 3, 4],
            CompactionShape {
                rows: 1,
                cols: 4,
                low: 0.5,
                high: 0.8,
            },
        )
        .unwrap();

        assert_eq!(output.row_counts(), &[2]);
        assert_eq!(output.row_values(0), Some(&[0.7, 0.5][..]));
        assert_eq!(output.row_indices(0), Some(&[2, 3][..]));
    }

    #[test]
    fn cpu_entrypoint_keeps_contract_failures_typed() {
        let error = compact_rows_cpu_f32(
            &[],
            &[],
            CompactionShape {
                rows: -1,
                cols: 0,
                low: 0.0,
                high: 1.0,
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            TensorCompactionError::Contract(CompactionError::NegativeDimensions { .. })
        ));
    }

    #[cfg(feature = "wgpu_dense")]
    #[test]
    fn wgpu_entrypoint_matches_cpu_when_enabled() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let Some((device, queue)) = test_device() else {
            eprintln!("skipping tensor compaction runtime test: no WGPU adapter");
            return;
        };
        let shape = CompactionShape {
            rows: 2,
            cols: 257,
            low: -1.5,
            high: 2.25,
        };
        let values = (0..514)
            .map(|index| ((index * 19 % 71) as f32 - 35.0) / 9.0)
            .collect::<Vec<_>>();
        let indices = (0..514).collect::<Vec<i32>>();
        let expected = compact_rows_cpu_f32(&values, &indices, shape).unwrap();
        let actual = compact_rows_wgpu_f32(&device, &queue, &values, &indices, shape).unwrap();
        assert_eq!(actual, expected);
    }

    #[cfg(feature = "wgpu_dense")]
    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("st.tensor.compaction.test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
            },
            None,
        ))
        .ok()
    }
}
