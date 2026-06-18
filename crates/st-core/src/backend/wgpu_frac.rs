// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Legacy WGPU fractional ops wrapper.
//!
//! This module is intentionally kept behind `wgpu-rt`: it is an older
//! st-core-local wrapper, but if it is revived it must fail as a structured
//! backend error rather than panicking inside a long training run.

use ndarray::{ArrayD, IxDyn};
use std::any::Any;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Duration;
use wgpu::util::DeviceExt;

use super::wgpu_rt;

const WGPU_FRAC_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum WgpuFracError {
    #[error("fractional WGPU input is empty or has a zero-sized selected axis")]
    EmptyInput,
    #[error("axis {axis} is out of bounds for tensor rank {rank}")]
    AxisOutOfBounds { axis: usize, rank: usize },
    #[error("{label} contained non-finite value at index {index}: {value}")]
    NonFiniteValue {
        label: &'static str,
        index: usize,
        value: f32,
    },
    #[error("{label} length overflow")]
    LengthOverflow { label: &'static str },
    #[error("shape error: {0}")]
    Shape(String),
    #[error("WGPU {stage} failed: {message}")]
    Backend {
        stage: &'static str,
        message: String,
    },
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        msg.to_string()
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        msg.clone()
    } else {
        "unknown panic".to_string()
    }
}

fn validate_finite_slice(label: &'static str, values: &[f32]) -> Result<(), WgpuFracError> {
    for (index, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(WgpuFracError::NonFiniteValue {
                label,
                index,
                value,
            });
        }
    }
    Ok(())
}

fn checked_len(label: &'static str, factors: &[usize]) -> Result<usize, WgpuFracError> {
    factors.iter().try_fold(1usize, |acc, factor| {
        acc.checked_mul(*factor)
            .ok_or(WgpuFracError::LengthOverflow { label })
    })
}

fn flatten_rows_cols(x: &ArrayD<f32>, axis: usize) -> Result<(Vec<f32>, u32, u32), WgpuFracError> {
    let rank = x.ndim();
    if axis >= rank {
        return Err(WgpuFracError::AxisOutOfBounds { axis, rank });
    }

    let mut perm: Vec<usize> = (0..rank).collect();
    perm.retain(|&dim| dim != axis);
    perm.push(axis);
    let xp = x.view().permuted_axes(perm);
    let cols = *xp.shape().last().ok_or(WgpuFracError::EmptyInput)?;
    if cols == 0 || xp.is_empty() {
        return Err(WgpuFracError::EmptyInput);
    }
    let rows = xp.len() / cols;
    let data: Vec<f32> = xp.iter().copied().collect();
    validate_finite_slice("fracdiff_input", &data)?;
    Ok((data, rows as u32, cols as u32))
}

fn create_shader(
    device: &wgpu::Device,
    label: &'static str,
    source: &'static str,
) -> Result<wgpu::ShaderModule, WgpuFracError> {
    catch_unwind(AssertUnwindSafe(|| {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }))
    .map_err(|payload| WgpuFracError::Backend {
        stage: "shader",
        message: panic_payload_to_string(payload),
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &'static str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
) -> Result<wgpu::ComputePipeline, WgpuFracError> {
    catch_unwind(AssertUnwindSafe(|| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module,
            entry_point: "main",
            compilation_options: Default::default(),
        })
    }))
    .map_err(|payload| WgpuFracError::Backend {
        stage: "pipeline",
        message: panic_payload_to_string(payload),
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FracParams {
    cols: u32,
    rows: u32,
    klen: u32,
    alpha_scale: f32,
    pad_mode: u32,
    _pad: [u32; 3],
}

/// Applies a Grünwald-Letnikov-style fractional difference with caller-supplied
/// coefficients. The output preserves the legacy axis-last `[rows, cols]`
/// wrapper shape instead of reconstructing the original ndarray layout.
pub fn fracdiff_gl_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x: &ArrayD<f32>,
    coeff: &[f32],
    alpha_scale: f32,
    axis: usize,
    pad_zero: bool,
) -> Result<ArrayD<f32>, WgpuFracError> {
    let (data, rows, cols) = flatten_rows_cols(x, axis)?;
    if coeff.is_empty() {
        return Err(WgpuFracError::EmptyInput);
    }
    validate_finite_slice("fracdiff_coeff", coeff)?;
    if !alpha_scale.is_finite() {
        return Err(WgpuFracError::NonFiniteValue {
            label: "fracdiff_alpha_scale",
            index: 0,
            value: alpha_scale,
        });
    }

    let n = checked_len("fracdiff_output", &[rows as usize, cols as usize])?;
    let byte_len = checked_len("fracdiff_output_bytes", &[n, std::mem::size_of::<f32>()])?;

    let buf_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fracdiff.X"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_c = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fracdiff.C"),
        contents: bytemuck::cast_slice(coeff),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let y = vec![0f32; n];
    let buf_y = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fracdiff.Y"),
        contents: bytemuck::cast_slice(&y),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let params = FracParams {
        cols,
        rows,
        klen: coeff.len() as u32,
        alpha_scale,
        pad_mode: if pad_zero { 0 } else { 1 },
        _pad: [0; 3],
    };
    let buf_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("fracdiff.Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = create_shader(
        device,
        "wgpu_fracdiff_gl.wgsl",
        include_str!("wgpu_shaders/wgpu_fracdiff_gl.wgsl"),
    )?;
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fracdiff.bind_layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fracdiff.pipeline_layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = create_pipeline(device, "fracdiff.pipeline", &pipeline_layout, &shader)?;
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fracdiff.bind"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_c.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_p.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fracdiff.encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fracdiff.pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind, &[]);
        cpass.dispatch_workgroups(rows.div_ceil(128), 1, 1);
    }

    let buf_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fracdiff.read"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_y, 0, &buf_read, 0, byte_len as u64);
    let cmd = encoder.finish();
    let cmd_bufs = [cmd];
    wgpu_rt::st_submit_with_timeout(device, queue, cmd_bufs, WGPU_FRAC_TIMEOUT).map_err(|err| {
        WgpuFracError::Backend {
            stage: "fracdiff_submit",
            message: err.to_string(),
        }
    })?;
    let bytes =
        wgpu_rt::st_map_read_with_timeout(device, &buf_read, 0..byte_len as u64, WGPU_FRAC_TIMEOUT)
            .map_err(|err| WgpuFracError::Backend {
                stage: "fracdiff_readback",
                message: err.to_string(),
            })?;
    let out: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    validate_finite_slice("fracdiff_output", &out)?;

    ArrayD::from_shape_vec(IxDyn(&[rows as usize, cols as usize]), out)
        .map_err(|err| WgpuFracError::Shape(err.to_string()))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpecParams {
    cols: u32,
    rows: u32,
    power: f32,
    _pad: u32,
}

/// Multiplies interleaved FFT coefficients by a simple fractional-Laplacian
/// frequency envelope. `x_fft_reim` must have length `rows * cols * 2`.
pub fn specmul_frac_laplace_wgpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    x_fft_reim: &[f32],
    rows: u32,
    cols: u32,
    power: f32,
) -> Result<Vec<f32>, WgpuFracError> {
    if rows == 0 || cols == 0 {
        return Err(WgpuFracError::EmptyInput);
    }
    if !power.is_finite() {
        return Err(WgpuFracError::NonFiniteValue {
            label: "specmul_power",
            index: 0,
            value: power,
        });
    }
    validate_finite_slice("specmul_input", x_fft_reim)?;
    let n_cplx = checked_len("specmul_complex_len", &[rows as usize, cols as usize])?;
    let expected_len = checked_len("specmul_input_len", &[n_cplx, 2])?;
    if x_fft_reim.len() != expected_len {
        return Err(WgpuFracError::Shape(format!(
            "expected interleaved FFT length {expected_len}, got {}",
            x_fft_reim.len()
        )));
    }
    let byte_len = checked_len(
        "specmul_output_bytes",
        &[expected_len, std::mem::size_of::<f32>()],
    )?;

    let buf_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("specmul.X"),
        contents: bytemuck::cast_slice(x_fft_reim),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let y = vec![0f32; expected_len];
    let buf_y = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("specmul.Y"),
        contents: bytemuck::cast_slice(&y),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    });
    let params = SpecParams {
        cols,
        rows,
        power,
        _pad: 0,
    };
    let buf_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("specmul.Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let shader = create_shader(
        device,
        "wgpu_frac_specmul.wgsl",
        include_str!("wgpu_shaders/wgpu_frac_specmul.wgsl"),
    )?;
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("specmul.bind_layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, false),
            uniform_entry(2),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("specmul.pipeline_layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let pipeline = create_pipeline(device, "specmul.pipeline", &pipeline_layout, &shader)?;
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("specmul.bind"),
        layout: &bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_p.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("specmul.encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("specmul.pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind, &[]);
        cpass.dispatch_workgroups(rows.div_ceil(16), cols.div_ceil(16), 1);
    }

    let buf_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("specmul.read"),
        size: byte_len as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_y, 0, &buf_read, 0, byte_len as u64);
    let cmd = encoder.finish();
    let cmd_bufs = [cmd];
    wgpu_rt::st_submit_with_timeout(device, queue, cmd_bufs, WGPU_FRAC_TIMEOUT).map_err(|err| {
        WgpuFracError::Backend {
            stage: "specmul_submit",
            message: err.to_string(),
        }
    })?;
    let bytes =
        wgpu_rt::st_map_read_with_timeout(device, &buf_read, 0..byte_len as u64, WGPU_FRAC_TIMEOUT)
            .map_err(|err| WgpuFracError::Backend {
                stage: "specmul_readback",
                message: err.to_string(),
            })?;
    let out: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    validate_finite_slice("specmul_output", &out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_rows_cols_rejects_invalid_axis() {
        let input = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let err = flatten_rows_cols(&input, 2).unwrap_err();
        assert_eq!(err, WgpuFracError::AxisOutOfBounds { axis: 2, rank: 2 });
    }

    #[test]
    fn flatten_rows_cols_rejects_non_finite_input() {
        let input = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![1.0, f32::NAN]).unwrap();
        let err = flatten_rows_cols(&input, 1).unwrap_err();
        assert!(matches!(
            err,
            WgpuFracError::NonFiniteValue {
                label: "fracdiff_input",
                index: 1,
                value,
            } if value.is_nan()
        ));
    }

    #[test]
    fn specmul_preflight_rejects_shape_and_non_finite_values_before_gpu_work() {
        validate_finite_slice("specmul_input", &[1.0, 2.0]).unwrap();
        let err = validate_finite_slice("specmul_input", &[1.0, f32::INFINITY]).unwrap_err();
        assert_eq!(
            err,
            WgpuFracError::NonFiniteValue {
                label: "specmul_input",
                index: 1,
                value: f32::INFINITY,
            }
        );
    }
}
