// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright 2025 Ryo SpiralArchitect

//! Exact two-command rank-k execution for WGPU.
//!
//! The first command sorts finite candidates inside planner-sized row tiles.
//! The second command merges those ordered runs and emits the exact TopK,
//! MidK, or BottomK result. Ordering and padding intentionally mirror the
//! Rust CPU reference contract: non-finite values are ignored, equal values
//! prefer the lower source index, and missing outputs are `(NaN, -1)`.

use std::any::Any;
use std::borrow::Cow;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc::{self, TryRecvError};
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use thiserror::Error;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;
const STORAGE_BINDINGS: u32 = 7;
const READBACK_TIMEOUT: Duration = Duration::from_secs(30);
const WGSL: &str = include_str!("shaders/rankk_exact_2ce.wgsl");

/// Exact rank family selected by the two-command dispatcher.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    TopK,
    MidK,
    BottomK,
}

impl Kind {
    const fn as_uniform(self) -> u32 {
        match self {
            Self::TopK => 0,
            Self::MidK => 1,
            Self::BottomK => 2,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TopK => "topk",
            Self::MidK => "midk",
            Self::BottomK => "bottomk",
        }
    }
}

/// Validated shape and tile geometry for exact two-command rank-k execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Plan {
    kind: Kind,
    rows: u32,
    cols: u32,
    k: u32,
    requested_tile_cols: u32,
    tile_cols: u32,
    tile_stride: u32,
    tiles_x: u32,
    input_elements: u32,
    output_elements: u32,
    scratch_elements: u32,
    tile_state_elements: u32,
}

impl Plan {
    pub fn try_new(
        kind: Kind,
        rows: u32,
        cols: u32,
        k: u32,
        tile_cols: u32,
    ) -> Result<Self, PlanError> {
        if tile_cols == 0 {
            return Err(PlanError::ZeroTile);
        }
        if k > cols {
            return Err(PlanError::KExceedsColumns { k, cols });
        }

        // A tile larger than the row is a valid one-tile request. Keep both
        // values so reports retain the planner directive and execution truth.
        let effective_tile_cols = tile_cols.min(cols.max(1));
        let tile_stride = effective_tile_cols
            .checked_next_power_of_two()
            .ok_or(PlanError::ArithmeticOverflow("tile stride"))?;
        let tiles_x = if cols == 0 {
            0
        } else {
            cols.div_ceil(effective_tile_cols)
        };

        let input_elements = checked_shader_elements("input", rows as u64 * cols as u64)?;
        let output_elements = checked_shader_elements("output", rows as u64 * k as u64)?;
        let tile_state_elements =
            checked_shader_elements("tile state", rows as u64 * tiles_x as u64)?;
        let scratch_elements = checked_shader_elements(
            "tile scratch",
            tile_state_elements as u64 * tile_stride as u64,
        )?;

        Ok(Self {
            kind,
            rows,
            cols,
            k,
            requested_tile_cols: tile_cols,
            tile_cols: effective_tile_cols,
            tile_stride,
            tiles_x,
            input_elements,
            output_elements,
            scratch_elements,
            tile_state_elements,
        })
    }

    pub const fn kind(self) -> Kind {
        self.kind
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn cols(self) -> u32 {
        self.cols
    }

    pub const fn k(self) -> u32 {
        self.k
    }

    pub const fn requested_tile_cols(self) -> u32 {
        self.requested_tile_cols
    }

    pub const fn tile_cols(self) -> u32 {
        self.tile_cols
    }

    pub const fn tile_stride(self) -> u32 {
        self.tile_stride
    }

    pub const fn tiles_x(self) -> u32 {
        self.tiles_x
    }

    pub const fn is_empty(self) -> bool {
        self.rows == 0 || self.cols == 0 || self.k == 0
    }

    pub const fn input_elements(self) -> u32 {
        self.input_elements
    }

    pub const fn output_elements(self) -> u32 {
        self.output_elements
    }

    pub const fn scratch_elements(self) -> u32 {
        self.scratch_elements
    }

    pub const fn tile_state_elements(self) -> u32 {
        self.tile_state_elements
    }

    fn params(self) -> ParamsUniform {
        ParamsUniform {
            rows: self.rows,
            cols: self.cols,
            k: self.k,
            tile_cols: self.tile_cols,
            tile_stride: self.tile_stride,
            tiles_x: self.tiles_x,
            kind: self.kind.as_uniform(),
            _pad: 0,
        }
    }

    fn validate_device(self, device: &wgpu::Device) -> Result<(), DispatchError> {
        if self.is_empty() {
            return Ok(());
        }

        let limits = device.limits();
        if WORKGROUP_SIZE > limits.max_compute_invocations_per_workgroup
            || WORKGROUP_SIZE > limits.max_compute_workgroup_size_x
        {
            return Err(DispatchError::DeviceLimit {
                resource: "workgroup_size_x",
                required: WORKGROUP_SIZE as u64,
                available: limits
                    .max_compute_invocations_per_workgroup
                    .min(limits.max_compute_workgroup_size_x) as u64,
            });
        }
        if STORAGE_BINDINGS > limits.max_storage_buffers_per_shader_stage {
            return Err(DispatchError::DeviceLimit {
                resource: "storage_buffers_per_shader_stage",
                required: STORAGE_BINDINGS as u64,
                available: limits.max_storage_buffers_per_shader_stage as u64,
            });
        }
        for (axis, required) in [("tiles_x", self.tiles_x), ("rows", self.rows)] {
            if required > limits.max_compute_workgroups_per_dimension {
                return Err(DispatchError::DeviceLimit {
                    resource: axis,
                    required: required as u64,
                    available: limits.max_compute_workgroups_per_dimension as u64,
                });
            }
        }

        let max_storage = u64::from(limits.max_storage_buffer_binding_size);
        let max_buffer = limits.max_buffer_size;
        for (name, elements) in [
            ("input", self.input_elements),
            ("output values", self.output_elements),
            ("output indices", self.output_elements),
            ("scratch values", self.scratch_elements),
            ("scratch indices", self.scratch_elements),
            ("tile counts", self.tile_state_elements),
            ("tile cursors", self.tile_state_elements),
        ] {
            let bytes = u64::from(elements) * 4;
            let available = max_storage.min(max_buffer);
            if bytes > available {
                return Err(DispatchError::DeviceLimit {
                    resource: name,
                    required: bytes,
                    available,
                });
            }
        }
        Ok(())
    }
}

fn checked_shader_elements(name: &'static str, elements: u64) -> Result<u32, PlanError> {
    u32::try_from(elements).map_err(|_| PlanError::IndexSpaceTooLarge { name, elements })
}

#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum PlanError {
    #[error("rank-k two-command execution requires tile_cols > 0")]
    ZeroTile,
    #[error("rank-k requires k <= cols, got k={k} cols={cols}")]
    KExceedsColumns { k: u32, cols: u32 },
    #[error("rank-k two-command geometry overflowed while computing {0}")]
    ArithmeticOverflow(&'static str),
    #[error("{name} requires {elements} elements, exceeding the WGSL u32 index space")]
    IndexSpaceTooLarge { name: &'static str, elements: u64 },
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ParamsUniform {
    rows: u32,
    cols: u32,
    k: u32,
    tile_cols: u32,
    tile_stride: u32,
    tiles_x: u32,
    kind: u32,
    _pad: u32,
}

/// Device-owned pipelines shared by repeated exact rank-k dispatches.
#[derive(Debug)]
pub struct Pipelines {
    layout: wgpu::BindGroupLayout,
    tile_sort: wgpu::ComputePipeline,
    row_merge: wgpu::ComputePipeline,
}

impl Pipelines {
    pub fn new(device: &wgpu::Device) -> Result<Self, DispatchError> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = catch_unwind(AssertUnwindSafe(|| {
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.rankk.exact_2ce.bind_group_layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, false),
                    storage_entry(2, false),
                    storage_entry(3, false),
                    storage_entry(4, false),
                    storage_entry(5, false),
                    storage_entry(6, false),
                    uniform_entry(7),
                ],
            });
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.rankk.exact_2ce.pipeline_layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("st.rankk.exact_2ce.shader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(WGSL)),
            });
            let tile_sort = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("st.rankk.exact_2ce.tile_sort"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "rankk_exact_2ce_tile_sort",
                compilation_options: Default::default(),
            });
            let row_merge = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("st.rankk.exact_2ce.row_merge"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "rankk_exact_2ce_row_merge",
                compilation_options: Default::default(),
            });
            Self {
                layout,
                tile_sort,
                row_merge,
            }
        }));
        let validation = pollster::block_on(device.pop_error_scope());

        match (built, validation) {
            (Err(payload), _) => Err(DispatchError::PipelineBuild(panic_payload_to_string(
                payload,
            ))),
            (_, Some(error)) => Err(DispatchError::PipelineBuild(error.to_string())),
            (Ok(pipelines), None) => Ok(pipelines),
        }
    }
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

/// Host-visible exact rank-k outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub values: Vec<f32>,
    pub indices: Vec<i32>,
}

/// Execute the validated plan as two ordered WGPU command submissions.
pub fn dispatch_host(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipelines: &Pipelines,
    plan: Plan,
    input: &[f32],
) -> Result<Output, DispatchError> {
    let expected = plan.input_elements as usize;
    if input.len() != expected {
        return Err(DispatchError::InputLength {
            expected,
            actual: input.len(),
        });
    }
    if plan.is_empty() {
        return Ok(Output {
            values: Vec::new(),
            indices: Vec::new(),
        });
    }
    plan.validate_device(device)?;

    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.rankk.exact_2ce.input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scratch_values = storage_buffer(
        device,
        "st.rankk.exact_2ce.scratch_values",
        plan.scratch_elements,
        false,
    );
    let scratch_indices = storage_buffer(
        device,
        "st.rankk.exact_2ce.scratch_indices",
        plan.scratch_elements,
        false,
    );
    let tile_counts = storage_buffer(
        device,
        "st.rankk.exact_2ce.tile_counts",
        plan.tile_state_elements,
        false,
    );
    let tile_cursors = storage_buffer(
        device,
        "st.rankk.exact_2ce.tile_cursors",
        plan.tile_state_elements,
        false,
    );
    let output_values = storage_buffer(
        device,
        "st.rankk.exact_2ce.output_values",
        plan.output_elements,
        true,
    );
    let output_indices = storage_buffer(
        device,
        "st.rankk.exact_2ce.output_indices",
        plan.output_elements,
        true,
    );
    let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.rankk.exact_2ce.params"),
        contents: bytemuck::bytes_of(&plan.params()),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.rankk.exact_2ce.bind_group"),
        layout: &pipelines.layout,
        entries: &[
            binding(0, &input_buffer),
            binding(1, &scratch_values),
            binding(2, &scratch_indices),
            binding(3, &tile_counts),
            binding(4, &tile_cursors),
            binding(5, &output_values),
            binding(6, &output_indices),
            binding(7, &params),
        ],
    });

    let mut first = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.rankk.exact_2ce.command.tile_sort"),
    });
    {
        let mut pass = first.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.rankk.exact_2ce.pass.tile_sort"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.tile_sort);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(plan.tiles_x, plan.rows, 1);
    }
    queue.submit(Some(first.finish()));

    let output_bytes = u64::from(plan.output_elements) * 4;
    let values_staging = staging_buffer(device, "st.rankk.exact_2ce.values_staging", output_bytes);
    let indices_staging =
        staging_buffer(device, "st.rankk.exact_2ce.indices_staging", output_bytes);
    let mut second = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.rankk.exact_2ce.command.row_merge"),
    });
    {
        let mut pass = second.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.rankk.exact_2ce.pass.row_merge"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.row_merge);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(plan.rows, 1, 1);
    }
    second.copy_buffer_to_buffer(&output_values, 0, &values_staging, 0, output_bytes);
    second.copy_buffer_to_buffer(&output_indices, 0, &indices_staging, 0, output_bytes);
    queue.submit(Some(second.finish()));

    let value_bytes = map_read(device, &values_staging, output_bytes)?;
    let index_bytes = map_read(device, &indices_staging, output_bytes)?;
    let values = value_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte f32 chunk")))
        .collect();
    let indices = index_bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_ne_bytes(bytes.try_into().expect("four-byte u32 chunk")) as i32)
        .collect();
    Ok(Output { values, indices })
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    elements: u32,
    copy_src: bool,
) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE;
    if copy_src {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: u64::from(elements) * 4,
        usage,
        mapped_at_creation: false,
    })
}

fn staging_buffer(device: &wgpu::Device, label: &'static str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn map_read(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    bytes: u64,
) -> Result<Vec<u8>, DispatchError> {
    let slice = buffer.slice(0..bytes);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let started = Instant::now();
    loop {
        device.poll(wgpu::Maintain::Poll);
        match rx.try_recv() {
            Ok(result) => {
                result.map_err(|error| DispatchError::Map(error.to_string()))?;
                break;
            }
            Err(TryRecvError::Empty) if started.elapsed() < READBACK_TIMEOUT => {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(TryRecvError::Empty) => return Err(DispatchError::ReadbackTimeout),
            Err(TryRecvError::Disconnected) => {
                return Err(DispatchError::Map(
                    "WGPU map callback disconnected".to_string(),
                ));
            }
        }
    }
    let view = slice.get_mapped_range();
    let data = view.to_vec();
    drop(view);
    buffer.unmap();
    Ok(data)
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown WGPU pipeline panic".to_string()
    }
}

#[derive(Debug, Error)]
pub enum DispatchError {
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error("rank-k input length mismatch: expected {expected}, got {actual}")]
    InputLength { expected: usize, actual: usize },
    #[error("WGPU limit '{resource}' requires {required}, device provides {available}")]
    DeviceLimit {
        resource: &'static str,
        required: u64,
        available: u64,
    },
    #[error("failed to build exact rank-k two-command pipelines: {0}")]
    PipelineBuild(String),
    #[error("failed to map exact rank-k output: {0}")]
    Map(String),
    #[error("exact rank-k WGPU readback timed out after 30 seconds")]
    ReadbackTimeout,
}

#[cfg(test)]
mod tests {
    use super::*;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    #[test]
    fn plan_preserves_requested_tile_and_derives_exact_geometry() {
        let plan = Plan::try_new(Kind::MidK, 2, 513, 7, 1024).unwrap();
        assert_eq!(plan.requested_tile_cols(), 1024);
        assert_eq!(plan.tile_cols(), 513);
        assert_eq!(plan.tile_stride(), 1024);
        assert_eq!(plan.tiles_x(), 1);
        assert_eq!(plan.input_elements(), 1026);
        assert_eq!(plan.output_elements(), 14);
        assert_eq!(plan.scratch_elements(), 2048);
        assert_eq!(plan.tile_state_elements(), 2);

        let tiled = Plan::try_new(Kind::BottomK, 2, 513, 7, 256).unwrap();
        assert_eq!(tiled.tiles_x(), 3);
        assert_eq!(tiled.tile_stride(), 256);
        assert_eq!(tiled.scratch_elements(), 1536);
    }

    #[test]
    fn plan_rejects_semantically_invalid_requests() {
        assert_eq!(
            Plan::try_new(Kind::TopK, 1, 8, 2, 0),
            Err(PlanError::ZeroTile)
        );
        assert_eq!(
            Plan::try_new(Kind::TopK, 1, 8, 9, 4),
            Err(PlanError::KExceedsColumns { k: 9, cols: 8 })
        );
    }

    #[test]
    fn exact_rank_2ce_wgsl_parses_and_validates() {
        let module = naga::front::wgsl::parse_str(WGSL).expect("rank-k 2CE WGSL should parse");
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .expect("rank-k 2CE WGSL should validate");
    }

    #[test]
    fn exact_rank_2ce_runtime_matches_cpu_reference_when_enabled() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let Some((device, queue)) = test_device() else {
            eprintln!("skipping exact rank-k 2CE runtime test: no WGPU adapter");
            return;
        };
        let pipelines = Pipelines::new(&device).expect("exact rank-k 2CE pipelines");

        let rows = 2;
        let cols = 513;
        let k = 7;
        let mut input = (0..rows * cols)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) / 7.0)
            .collect::<Vec<_>>();
        input[0] = f32::NAN;
        input[1] = f32::INFINITY;
        input[2] = -0.0;
        input[3] = 0.0;
        input[129] = 9.25;
        input[385] = 9.25;
        for value in &mut input[cols as usize..] {
            *value = f32::NAN;
        }
        input[cols as usize + 2] = -0.0;
        input[cols as usize + 257] = 0.0;
        input[cols as usize + 512] = -4.0;

        for tile_cols in [70, 128, 257, 300, 1_024] {
            for kind in [Kind::TopK, Kind::MidK, Kind::BottomK] {
                let plan = Plan::try_new(kind, rows, cols, k, tile_cols).unwrap();
                let actual = dispatch_host(&device, &queue, &pipelines, plan, &input)
                    .unwrap_or_else(|error| {
                        panic!(
                            "{} 2CE tile={tile_cols} dispatch failed: {error}",
                            kind.as_str()
                        )
                    });
                let expected = cpu_reference(kind, rows, cols, k, &input);
                assert_eq!(
                    actual.indices,
                    expected.indices,
                    "{} tile={tile_cols} indices",
                    kind.as_str()
                );
                for (slot, (actual, expected)) in
                    actual.values.iter().zip(expected.values.iter()).enumerate()
                {
                    assert!(
                        (actual.is_nan() && expected.is_nan())
                            || actual.to_bits() == expected.to_bits(),
                        "{} tile={tile_cols} value slot {slot}: actual={actual:?} expected={expected:?}",
                        kind.as_str()
                    );
                }
            }
        }
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
                label: Some("st.rankk.exact_2ce.test_device"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
            },
            None,
        ))
        .ok()
    }

    fn cpu_reference(kind: Kind, rows: u32, cols: u32, k: u32, input: &[f32]) -> Output {
        let mut values = Vec::with_capacity((rows * k) as usize);
        let mut indices = Vec::with_capacity((rows * k) as usize);
        for row in 0..rows as usize {
            let row_values = &input[row * cols as usize..(row + 1) * cols as usize];
            let mut finite = (0..cols as usize)
                .filter(|&index| row_values[index].is_finite())
                .collect::<Vec<_>>();
            finite.sort_unstable_by(|&left, &right| {
                let order = match kind {
                    Kind::TopK => row_values[right].total_cmp(&row_values[left]),
                    Kind::MidK | Kind::BottomK => row_values[left].total_cmp(&row_values[right]),
                };
                order.then_with(|| left.cmp(&right))
            });
            let take = (k as usize).min(finite.len());
            let start = if kind == Kind::MidK {
                (finite.len() - take) / 2
            } else {
                0
            };
            for &index in &finite[start..start + take] {
                values.push(row_values[index]);
                indices.push(index as i32);
            }
            values.extend(std::iter::repeat_n(f32::NAN, k as usize - take));
            indices.extend(std::iter::repeat_n(-1, k as usize - take));
        }
        debug_assert_eq!(values.len(), (rows * k) as usize);
        debug_assert_eq!(indices.len(), (rows * k) as usize);
        Output { values, indices }
    }
}
