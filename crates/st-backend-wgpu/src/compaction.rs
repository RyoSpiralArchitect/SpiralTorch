// SPDX-License-Identifier: AGPL-3.0-or-later

//! Contract-driven stable row compaction for WGPU.
//!
//! Semantics come from `st-kernel-contracts`; this module owns only pipeline
//! construction, device validation, dispatch ordering, and host readback.

use std::any::Any;
use std::borrow::Cow;
use std::panic::{catch_unwind, AssertUnwindSafe};

use bytemuck::{Pod, Zeroable};
use st_kernel_contracts::compaction::{
    CompactionError, CompactionLayout, CompactionOutputF32, CompactionShape,
};
use thiserror::Error;

use crate::runtime::{
    empty_buffer, ensure_blocking_readback_supported, read_buffer, upload_slice, WgpuRuntimeError,
};

const WORKGROUP_SIZE: u32 = 256;
const MAX_STORAGE_BINDINGS: u32 = 6;
const SCAN_WGSL: &str = include_str!("shaders/wgpu_compaction_scan.wgsl");
const APPLY_WGSL: &str = include_str!("shaders/wgpu_compaction_apply.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    rows: u32,
    cols: u32,
    low: f32,
    high: f32,
    tiles_per_row: u32,
}

/// Pipelines reused by repeated stable compaction dispatches.
#[derive(Debug)]
pub struct CompactionPipelines {
    scan_layout: wgpu::BindGroupLayout,
    apply_layout: wgpu::BindGroupLayout,
    scan: wgpu::ComputePipeline,
    apply: wgpu::ComputePipeline,
}

impl CompactionPipelines {
    pub fn new(device: &wgpu::Device) -> Result<Self, CompactionDispatchError> {
        #[cfg(not(target_arch = "wasm32"))]
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = catch_unwind(AssertUnwindSafe(|| {
            let scan_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.compaction.scan.bind_group_layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    uniform_entry(3),
                ],
            });
            let apply_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.compaction.apply.bind_group_layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, false),
                    storage_entry(5, false),
                    uniform_entry(6),
                ],
            });
            let scan_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("st.compaction.scan.pipeline_layout"),
                    bind_group_layouts: &[&scan_layout],
                    push_constant_ranges: &[],
                });
            let apply_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("st.compaction.apply.pipeline_layout"),
                    bind_group_layouts: &[&apply_layout],
                    push_constant_ranges: &[],
                });
            let scan_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("st.compaction.scan.shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SCAN_WGSL)),
            });
            let apply_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("st.compaction.apply.shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(APPLY_WGSL)),
            });
            let scan = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("st.compaction.scan.pipeline"),
                layout: Some(&scan_pipeline_layout),
                module: &scan_module,
                entry_point: "main_cs",
                compilation_options: Default::default(),
            });
            let apply = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("st.compaction.apply.pipeline"),
                layout: Some(&apply_pipeline_layout),
                module: &apply_module,
                entry_point: "main_cs",
                compilation_options: Default::default(),
            });
            Self {
                scan_layout,
                apply_layout,
                scan,
                apply,
            }
        }));
        #[cfg(not(target_arch = "wasm32"))]
        let validation = pollster::block_on(device.pop_error_scope());
        #[cfg(not(target_arch = "wasm32"))]
        match (built, validation) {
            (Err(payload), _) => Err(CompactionDispatchError::PipelineBuild(
                panic_payload_to_string(payload),
            )),
            (_, Some(error)) => Err(CompactionDispatchError::PipelineBuild(error.to_string())),
            (Ok(pipelines), None) => Ok(pipelines),
        }
        #[cfg(target_arch = "wasm32")]
        match built {
            Err(payload) => Err(CompactionDispatchError::PipelineBuild(
                panic_payload_to_string(payload),
            )),
            Ok(pipelines) => Ok(pipelines),
        }
    }
}

/// Build pipelines and execute one contract-validated compaction operation.
pub fn compact_rows_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    values: &[f32],
    indices: &[i32],
    shape: CompactionShape,
) -> Result<CompactionOutputF32, CompactionDispatchError> {
    let layout = shape.layout()?;
    layout.validate_input_storage(values, indices)?;
    if layout.is_empty() {
        return Ok(CompactionOutputF32::zeroed(layout));
    }
    ensure_blocking_readback_supported("host-visible compaction")?;
    let pipelines = CompactionPipelines::new(device)?;
    dispatch_host(device, queue, &pipelines, values, indices, shape)
}

/// Execute stable compaction using already-built pipelines.
pub fn dispatch_host(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &CompactionPipelines,
    values: &[f32],
    indices: &[i32],
    shape: CompactionShape,
) -> Result<CompactionOutputF32, CompactionDispatchError> {
    let layout = shape.layout()?;
    layout.validate_input_storage(values, indices)?;
    if layout.is_empty() {
        return Ok(CompactionOutputF32::zeroed(layout));
    }
    ensure_blocking_readback_supported("host-visible compaction")?;
    validate_device(device, layout)?;

    let params = Params {
        rows: u32::try_from(layout.rows()).expect("contract rows fit u32"),
        cols: u32::try_from(layout.cols()).expect("contract columns fit u32"),
        low: shape.low,
        high: shape.high,
        tiles_per_row: u32::try_from(layout.tiles_per_row()).expect("contract tile count fits u32"),
    };
    let input_values = upload_slice(
        device,
        "st.compaction.input.values",
        values,
        wgpu::BufferUsages::STORAGE,
    )?;
    let input_indices = upload_slice(
        device,
        "st.compaction.input.indices",
        indices,
        wgpu::BufferUsages::STORAGE,
    )?;
    let flags = empty_buffer::<u32>(
        device,
        "st.compaction.flags",
        layout.element_count(),
        wgpu::BufferUsages::STORAGE,
    )?;
    let tile_counts = empty_buffer::<u32>(
        device,
        "st.compaction.tile_counts",
        layout.tile_count(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )?;
    let output_values = empty_buffer::<f32>(
        device,
        "st.compaction.output.values",
        layout.element_count(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )?;
    let output_indices = empty_buffer::<i32>(
        device,
        "st.compaction.output.indices",
        layout.element_count(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )?;
    let params_buffer = upload_slice(
        device,
        "st.compaction.params",
        std::slice::from_ref(&params),
        wgpu::BufferUsages::UNIFORM,
    )?;

    let scan_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.compaction.scan.bind_group"),
        layout: &pipelines.scan_layout,
        entries: &[
            binding(0, &input_values),
            binding(1, &flags),
            binding(2, &tile_counts),
            binding(3, &params_buffer),
        ],
    });
    let apply_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.compaction.apply.bind_group"),
        layout: &pipelines.apply_layout,
        entries: &[
            binding(0, &input_values),
            binding(1, &input_indices),
            binding(2, &flags),
            binding(3, &tile_counts),
            binding(4, &output_values),
            binding(5, &output_indices),
            binding(6, &params_buffer),
        ],
    });
    let workgroups =
        u32::try_from(layout.tile_count()).map_err(|_| CompactionDispatchError::DeviceLimit {
            resource: "workgroups_x",
            required: layout.tile_count() as u64,
            available: u64::from(device.limits().max_compute_workgroups_per_dimension),
        })?;

    let mut scan_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.compaction.scan.encoder"),
    });
    {
        let mut pass = scan_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.compaction.scan.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.scan);
        pass.set_bind_group(0, &scan_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(Some(scan_encoder.finish()));

    let mut apply_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.compaction.apply.encoder"),
    });
    apply_encoder.clear_buffer(&output_values, 0, None);
    apply_encoder.clear_buffer(&output_indices, 0, None);
    {
        let mut pass = apply_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.compaction.apply.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.apply);
        pass.set_bind_group(0, &apply_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    queue.submit(Some(apply_encoder.finish()));

    let values_host = read_buffer::<f32>(
        device,
        queue,
        &output_values,
        layout.element_count(),
        "st.compaction.output.values.readback",
    )?;
    let indices_host = read_buffer::<i32>(
        device,
        queue,
        &output_indices,
        layout.element_count(),
        "st.compaction.output.indices.readback",
    )?;
    let tile_counts_host = read_buffer::<u32>(
        device,
        queue,
        &tile_counts,
        layout.tile_count(),
        "st.compaction.tile_counts.readback",
    )?;
    let row_counts = row_counts_from_tiles(layout, &tile_counts_host)?;
    Ok(CompactionOutputF32::from_parts(
        layout,
        values_host,
        indices_host,
        row_counts,
    )?)
}

fn row_counts_from_tiles(
    layout: CompactionLayout,
    tile_counts: &[u32],
) -> Result<Vec<u32>, CompactionDispatchError> {
    if tile_counts.len() != layout.tile_count() {
        return Err(CompactionDispatchError::TileCountLength {
            expected: layout.tile_count(),
            actual: tile_counts.len(),
        });
    }
    tile_counts
        .chunks_exact(layout.tiles_per_row())
        .enumerate()
        .map(|(row, counts)| {
            counts.iter().try_fold(0u32, |total, &count| {
                total
                    .checked_add(count)
                    .ok_or(CompactionDispatchError::RowCountOverflow { row })
            })
        })
        .collect()
}

fn validate_device(
    device: &wgpu::Device,
    layout: CompactionLayout,
) -> Result<(), CompactionDispatchError> {
    let limits = device.limits();
    let available_workgroup = limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size_x);
    if WORKGROUP_SIZE > available_workgroup {
        return Err(CompactionDispatchError::DeviceLimit {
            resource: "workgroup_size_x",
            required: u64::from(WORKGROUP_SIZE),
            available: u64::from(available_workgroup),
        });
    }
    if MAX_STORAGE_BINDINGS > limits.max_storage_buffers_per_shader_stage {
        return Err(CompactionDispatchError::DeviceLimit {
            resource: "storage_buffers_per_shader_stage",
            required: u64::from(MAX_STORAGE_BINDINGS),
            available: u64::from(limits.max_storage_buffers_per_shader_stage),
        });
    }
    let max_workgroups = usize::try_from(limits.max_compute_workgroups_per_dimension)
        .expect("u32 workgroup limit fits usize");
    if layout.tile_count() > max_workgroups {
        return Err(CompactionDispatchError::DeviceLimit {
            resource: "workgroups_x",
            required: layout.tile_count() as u64,
            available: u64::from(limits.max_compute_workgroups_per_dimension),
        });
    }
    Ok(())
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

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown WGPU pipeline panic".to_owned()
    }
}

#[derive(Debug, Error)]
pub enum CompactionDispatchError {
    #[error(transparent)]
    Contract(#[from] CompactionError),
    #[error(transparent)]
    Runtime(#[from] WgpuRuntimeError),
    #[error("failed to build WGPU compaction pipelines: {0}")]
    PipelineBuild(String),
    #[error("WGPU compaction limit '{resource}' requires {required}, device provides {available}")]
    DeviceLimit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
    #[error("WGPU compaction tile count length mismatch: expected {expected}, got {actual}")]
    TileCountLength { expected: usize, actual: usize },
    #[error("WGPU compaction row {row} count overflowed u32")]
    RowCountOverflow { row: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_kernel_contracts::compaction::compact_rows_reference_f32;

    #[test]
    fn tile_counts_reduce_to_exact_rows() {
        let layout = CompactionShape {
            rows: 2,
            cols: 513,
            low: -1.0,
            high: 1.0,
        }
        .layout()
        .unwrap();
        assert_eq!(
            row_counts_from_tiles(layout, &[256, 200, 1, 0, 2, 1]).unwrap(),
            vec![457, 3]
        );
    }

    #[test]
    fn runtime_matches_shared_reference_when_enabled() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let Some((device, queue)) = test_device() else {
            eprintln!("skipping compaction runtime test: no WGPU adapter");
            return;
        };
        let shape = CompactionShape {
            rows: 2,
            cols: 513,
            low: -2.5,
            high: 3.0,
        };
        let mut values = (0..1026)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) / 7.0)
            .collect::<Vec<_>>();
        values[0] = f32::NAN;
        values[512] = f32::INFINITY;
        let indices = (0..1026).collect::<Vec<i32>>();
        let expected = compact_rows_reference_f32(&values, &indices, shape).unwrap();
        let actual = compact_rows_f32(&device, &queue, &values, &indices, shape).unwrap();
        assert_eq!(actual, expected);
    }

    fn test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("st.compaction.test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
            },
            None,
        ))
        .ok()
    }
}
