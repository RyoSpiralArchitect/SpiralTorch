// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[cfg(not(target_arch = "wasm32"))]
use crate::runtime::ensure_default_runtime_blocking;
use crate::{
    runtime::{
        empty_buffer, ensure_blocking_readback_supported, read_buffer, upload_slice, Shared,
        WgpuContext, WgpuRuntimeError,
    },
    ShaderCache, ShaderLoadError,
};
use bytemuck::{Pod, Zeroable};
use std::path::{Path, PathBuf};
use thiserror::Error;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferUsages, ComputePipeline, Device,
    PipelineLayoutDescriptor, Queue, ShaderStages,
};

#[cfg(not(target_arch = "wasm32"))]
const TRANSFORM_SHADER_DIR: &str = "shaders/transforms";

#[derive(Debug, Error)]
pub enum TransformDispatchError {
    #[error(transparent)]
    Shader(#[from] ShaderLoadError),
    #[error("invalid transform geometry: {0}")]
    InvalidGeometry(String),
    #[error(transparent)]
    Runtime(#[from] WgpuRuntimeError),
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResizeParams {
    src_height: u32,
    src_width: u32,
    dst_height: u32,
    dst_width: u32,
    channels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CropParams {
    src_height: u32,
    src_width: u32,
    dst_height: u32,
    dst_width: u32,
    top: u32,
    left: u32,
    channels: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FlipParams {
    height: u32,
    width: u32,
    channels: u32,
    apply: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ColorJitterParams {
    dims: [u32; 4],
    factors: [f32; 4],
    means: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct ResizeConfig {
    pub channels: usize,
    pub src_height: usize,
    pub src_width: usize,
    pub dst_height: usize,
    pub dst_width: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct CenterCropConfig {
    pub channels: usize,
    pub src_height: usize,
    pub src_width: usize,
    pub crop_height: usize,
    pub crop_width: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct HorizontalFlipConfig {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub apply: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct ColorJitterConfig {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
}

struct Pipelines {
    bind_layout: BindGroupLayout,
    resize: Shared<ComputePipeline>,
    center_crop: Shared<ComputePipeline>,
    horizontal_flip: Shared<ComputePipeline>,
    color_jitter: Shared<ComputePipeline>,
}

impl Pipelines {
    fn new(device: &Device, shader_dir: impl AsRef<Path>) -> Result<Self, ShaderLoadError> {
        let bind_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("st.backend.transform.bind_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("st.backend.transform.pipeline_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let shader_root = shader_dir.as_ref();
        let cache = ShaderCache::new(shader_root);
        cache.prefetch([
            "resize.wgsl",
            "center_crop.wgsl",
            "horizontal_flip.wgsl",
            "color_jitter.wgsl",
        ])?;

        let resize = cache.load_compute_pipeline_with_layout(
            device,
            "resize.wgsl",
            "st.transform.resize",
            "main",
            Some(&pipeline_layout),
        )?;
        let center_crop = cache.load_compute_pipeline_with_layout(
            device,
            "center_crop.wgsl",
            "st.transform.center_crop",
            "main",
            Some(&pipeline_layout),
        )?;
        let horizontal_flip = cache.load_compute_pipeline_with_layout(
            device,
            "horizontal_flip.wgsl",
            "st.transform.horizontal_flip",
            "main",
            Some(&pipeline_layout),
        )?;
        let color_jitter = cache.load_compute_pipeline_with_layout(
            device,
            "color_jitter.wgsl",
            "st.transform.color_jitter",
            "main",
            Some(&pipeline_layout),
        )?;

        Ok(Self {
            bind_layout,
            resize,
            center_crop,
            horizontal_flip,
            color_jitter,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageGeometry {
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl ImageGeometry {
    pub fn element_count(&self) -> Result<usize, TransformDispatchError> {
        self.channels
            .checked_mul(self.height)
            .and_then(|v| v.checked_mul(self.width))
            .ok_or_else(|| {
                TransformDispatchError::InvalidGeometry(format!(
                    "image volume overflow for {}x{}x{}",
                    self.channels, self.height, self.width
                ))
            })
    }
}

#[derive(Clone, Copy, Debug)]
pub enum GeometryCommand {
    Resize(ResizeConfig),
    CenterCrop(CenterCropConfig),
    HorizontalFlip(HorizontalFlipConfig),
}

struct GpuContext {
    context: WgpuContext,
    pipelines: Pipelines,
}

impl GpuContext {
    fn bind_group(&self, input: &Buffer, output: &Buffer, params: &Buffer) -> BindGroup {
        self.context
            .device()
            .create_bind_group(&BindGroupDescriptor {
                label: Some("st.backend.transform.bind_group"),
                layout: &self.pipelines.bind_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: input.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: output.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            })
    }
}

enum Backend {
    Cpu,
    Gpu(GpuContext),
}

pub struct TransformDispatcher {
    backend: Backend,
}

impl TransformDispatcher {
    pub fn cpu() -> Self {
        Self {
            backend: Backend::Cpu,
        }
    }

    pub fn with_gpu(
        device: Shared<Device>,
        queue: Shared<Queue>,
        shader_dir: impl Into<PathBuf>,
    ) -> Result<Self, TransformDispatchError> {
        let pipelines = Pipelines::new(device.as_ref(), shader_dir.into())?;
        Ok(Self {
            backend: Backend::Gpu(GpuContext {
                context: WgpuContext::new(device, queue),
                pipelines,
            }),
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_default_gpu() -> Result<Self, TransformDispatchError> {
        let (runtime, _) = ensure_default_runtime_blocking("st.backend.transform.device")?;
        let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TRANSFORM_SHADER_DIR);
        Self::with_gpu(
            runtime.context().shared_device(),
            runtime.context().shared_queue(),
            shader_dir,
        )
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_default_gpu() -> Result<Self, TransformDispatchError> {
        Err(WgpuRuntimeError::UnsupportedTarget {
            operation: "synchronous adapter acquisition",
            target: "wasm32",
        }
        .into())
    }

    fn validate_volume(
        label: &str,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<usize, TransformDispatchError> {
        if channels == 0 || height == 0 || width == 0 {
            return Err(TransformDispatchError::InvalidGeometry(format!(
                "{label} dimensions must be positive, got {channels}x{height}x{width}"
            )));
        }
        for (field, value) in [("channels", channels), ("height", height), ("width", width)] {
            u32::try_from(value).map_err(|_| {
                TransformDispatchError::InvalidGeometry(format!(
                    "{label} {field} {value} exceeds the WGSL u32 range"
                ))
            })?;
        }
        channels
            .checked_mul(height)
            .and_then(|count| count.checked_mul(width))
            .ok_or_else(|| {
                TransformDispatchError::InvalidGeometry(format!(
                    "{label} volume overflow for {channels}x{height}x{width}"
                ))
            })
    }

    fn validate_finite(label: &str, value: f32) -> Result<(), TransformDispatchError> {
        if !value.is_finite() {
            return Err(TransformDispatchError::InvalidGeometry(format!(
                "{label} must be finite, got {value}"
            )));
        }
        Ok(())
    }

    pub fn resize(
        &self,
        input: &[f32],
        config: ResizeConfig,
    ) -> Result<Vec<f32>, TransformDispatchError> {
        let expected = Self::validate_volume(
            "resize source",
            config.channels,
            config.src_height,
            config.src_width,
        )?;
        Self::validate_volume(
            "resize destination",
            config.channels,
            config.dst_height,
            config.dst_width,
        )?;
        if input.len() != expected {
            return Err(TransformDispatchError::InvalidGeometry(format!(
                "input length {} does not match {}x{}x{}",
                input.len(),
                config.channels,
                config.src_height,
                config.src_width
            )));
        }
        match &self.backend {
            Backend::Cpu => Ok(cpu_resize(input, config)),
            Backend::Gpu(ctx) => gpu_resize(ctx, input, config),
        }
    }

    pub fn center_crop(
        &self,
        input: &[f32],
        config: CenterCropConfig,
    ) -> Result<Vec<f32>, TransformDispatchError> {
        let expected = Self::validate_volume(
            "center crop source",
            config.channels,
            config.src_height,
            config.src_width,
        )?;
        Self::validate_volume(
            "center crop destination",
            config.channels,
            config.crop_height,
            config.crop_width,
        )?;
        if config.crop_height > config.src_height || config.crop_width > config.src_width {
            return Err(TransformDispatchError::InvalidGeometry(
                "crop must fit inside source".into(),
            ));
        }
        if input.len() != expected {
            return Err(TransformDispatchError::InvalidGeometry(format!(
                "input length {} does not match {}x{}x{}",
                input.len(),
                config.channels,
                config.src_height,
                config.src_width
            )));
        }
        match &self.backend {
            Backend::Cpu => Ok(cpu_center_crop(input, config)),
            Backend::Gpu(ctx) => gpu_center_crop(ctx, input, config),
        }
    }

    pub fn horizontal_flip(
        &self,
        input: &[f32],
        config: HorizontalFlipConfig,
    ) -> Result<Vec<f32>, TransformDispatchError> {
        let expected = Self::validate_volume(
            "horizontal flip",
            config.channels,
            config.height,
            config.width,
        )?;
        if input.len() != expected {
            return Err(TransformDispatchError::InvalidGeometry(format!(
                "input length {} does not match {}x{}x{}",
                input.len(),
                config.channels,
                config.height,
                config.width
            )));
        }
        match &self.backend {
            Backend::Cpu => Ok(cpu_horizontal_flip(input, config)),
            Backend::Gpu(ctx) => gpu_horizontal_flip(ctx, input, config),
        }
    }

    pub fn color_jitter(
        &self,
        input: &[f32],
        config: ColorJitterConfig,
    ) -> Result<Vec<f32>, TransformDispatchError> {
        let expected =
            Self::validate_volume("color jitter", config.channels, config.height, config.width)?;
        for (label, value) in [
            ("brightness", config.brightness),
            ("contrast", config.contrast),
            ("saturation", config.saturation),
            ("hue", config.hue),
        ] {
            Self::validate_finite(label, value)?;
        }
        if input.len() != expected {
            return Err(TransformDispatchError::InvalidGeometry(format!(
                "input length {} does not match {}x{}x{}",
                input.len(),
                config.channels,
                config.height,
                config.width
            )));
        }
        let mut means = [0.0f32; 4];
        if config.contrast != 1.0 {
            let pixels = config.height * config.width;
            for (c, mean) in means.iter_mut().enumerate().take(config.channels.min(4)) {
                let start = c * pixels;
                let end = start + pixels;
                let slice = &input[start..end];
                *mean = slice.iter().sum::<f32>() / pixels as f32;
            }
        }
        match &self.backend {
            Backend::Cpu => Ok(cpu_color_jitter(input, config, means)),
            Backend::Gpu(ctx) => gpu_color_jitter(ctx, input, config, means),
        }
    }

    pub fn run_geometry_sequence(
        &self,
        input: &[f32],
        initial: ImageGeometry,
        commands: &[GeometryCommand],
    ) -> Result<(Vec<f32>, ImageGeometry), TransformDispatchError> {
        Self::validate_volume(
            "initial image",
            initial.channels,
            initial.height,
            initial.width,
        )?;
        let expected = initial.element_count()?;
        if input.len() != expected {
            return Err(TransformDispatchError::InvalidGeometry(format!(
                "input length {} does not match geometry {}x{}x{}",
                input.len(),
                initial.channels,
                initial.height,
                initial.width
            )));
        }
        if commands.is_empty() {
            return Ok((input.to_vec(), initial));
        }

        let mut geometries = Vec::with_capacity(commands.len());
        let mut current = initial;
        for command in commands {
            current = validate_geometry_transition(current, *command)?;
            geometries.push(current);
        }

        match &self.backend {
            Backend::Cpu => {
                let mut data = input.to_vec();
                for (command, geometry) in commands.iter().zip(&geometries) {
                    data = match *command {
                        GeometryCommand::Resize(config) => cpu_resize(&data, config),
                        GeometryCommand::CenterCrop(config) => cpu_center_crop(&data, config),
                        GeometryCommand::HorizontalFlip(config) => {
                            cpu_horizontal_flip(&data, config)
                        }
                    };
                    debug_assert_eq!(data.len(), geometry.element_count()?);
                }
                let final_geometry = *geometries.last().unwrap_or(&initial);
                Ok((data, final_geometry))
            }
            Backend::Gpu(ctx) => {
                ensure_blocking_readback_supported("host-visible transform sequence")?;
                let device = ctx.context.device();
                let queue = ctx.context.queue();
                let mut current_buffer = upload_slice(
                    device,
                    "st.backend.transform.sequence.input",
                    input,
                    BufferUsages::STORAGE,
                )?;
                for (command, _geometry) in commands.iter().zip(&geometries) {
                    let next_buffer = match *command {
                        GeometryCommand::Resize(config) => {
                            dispatch_resize_buffer(ctx, &current_buffer, config)?
                        }
                        GeometryCommand::CenterCrop(config) => {
                            dispatch_center_crop_buffer(ctx, &current_buffer, config)?
                        }
                        GeometryCommand::HorizontalFlip(config) => {
                            dispatch_horizontal_flip_buffer(ctx, &current_buffer, config)?
                        }
                    };
                    current_buffer = next_buffer;
                }
                let final_geometry = *geometries.last().unwrap();
                let output = read_buffer(
                    device,
                    queue,
                    &current_buffer,
                    final_geometry.element_count()?,
                    "st.backend.transform.sequence.readback",
                )?;
                Ok((output, final_geometry))
            }
        }
    }
}

fn workgroup_dims(
    device: &Device,
    width: usize,
    height: usize,
    depth: usize,
    x: u32,
    y: u32,
    z: u32,
) -> Result<(u32, u32, u32), TransformDispatchError> {
    let width = shader_dimension("width", width)?;
    let height = shader_dimension("height", height)?;
    let depth = shader_dimension("channels", depth)?;
    let geometry = (width.div_ceil(x), height.div_ceil(y), depth.div_ceil(z));
    let available = device.limits().max_compute_workgroups_per_dimension;
    for (axis, required) in [("x", geometry.0), ("y", geometry.1), ("z", geometry.2)] {
        if required > available {
            return Err(TransformDispatchError::InvalidGeometry(format!(
                "transform dispatch axis {axis} requires {required} workgroups, device provides {available}"
            )));
        }
    }
    Ok(geometry)
}

fn shader_dimension(field: &str, value: usize) -> Result<u32, TransformDispatchError> {
    u32::try_from(value).map_err(|_| {
        TransformDispatchError::InvalidGeometry(format!(
            "transform {field} {value} exceeds the WGSL u32 range"
        ))
    })
}

fn cpu_resize(input: &[f32], config: ResizeConfig) -> Vec<f32> {
    let mut output = vec![0.0f32; config.channels * config.dst_height * config.dst_width];
    let scale_y = config.src_height as f32 / config.dst_height as f32;
    let scale_x = config.src_width as f32 / config.dst_width as f32;
    let src_stride = config.src_width;
    let dst_stride = config.dst_width;
    let src_height = config.src_height;
    let src_width = config.src_width;
    let plane = config.src_height * config.src_width;
    for c in 0..config.channels {
        for y in 0..config.dst_height {
            let src_y = (y as f32 + 0.5) * scale_y - 0.5;
            let y0 = src_y.floor().clamp(0.0, (src_height - 1) as f32) as usize;
            let y1 = (y0 + 1).min(src_height - 1);
            let ly = src_y - y0 as f32;
            for x in 0..config.dst_width {
                let src_x = (x as f32 + 0.5) * scale_x - 0.5;
                let x0 = src_x.floor().clamp(0.0, (src_width - 1) as f32) as usize;
                let x1 = (x0 + 1).min(src_width - 1);
                let lx = src_x - x0 as f32;
                let base = c * plane;
                let top_left = input[base + y0 * src_stride + x0];
                let top_right = input[base + y0 * src_stride + x1];
                let bottom_left = input[base + y1 * src_stride + x0];
                let bottom_right = input[base + y1 * src_stride + x1];
                let top = top_left * (1.0 - lx) + top_right * lx;
                let bottom = bottom_left * (1.0 - lx) + bottom_right * lx;
                let value = top * (1.0 - ly) + bottom * ly;
                output[c * config.dst_height * dst_stride + y * dst_stride + x] = value;
            }
        }
    }
    output
}

fn cpu_center_crop(input: &[f32], config: CenterCropConfig) -> Vec<f32> {
    let mut output = vec![0.0f32; config.channels * config.crop_height * config.crop_width];
    let top = (config.src_height - config.crop_height) / 2;
    let left = (config.src_width - config.crop_width) / 2;
    let src_plane = config.src_height * config.src_width;
    let dst_plane = config.crop_height * config.crop_width;
    for c in 0..config.channels {
        for y in 0..config.crop_height {
            for x in 0..config.crop_width {
                let src_idx = c * src_plane + (top + y) * config.src_width + (left + x);
                let dst_idx = c * dst_plane + y * config.crop_width + x;
                output[dst_idx] = input[src_idx];
            }
        }
    }
    output
}

fn cpu_horizontal_flip(input: &[f32], config: HorizontalFlipConfig) -> Vec<f32> {
    if !config.apply {
        return input.to_vec();
    }
    let mut output = input.to_vec();
    let plane = config.height * config.width;
    for c in 0..config.channels {
        for y in 0..config.height {
            for x in 0..(config.width / 2) {
                let left = c * plane + y * config.width + x;
                let right = c * plane + y * config.width + (config.width - 1 - x);
                output.swap(left, right);
            }
        }
    }
    output
}

fn apply_contrast(slice: &mut [f32], mean: f32, factor: f32) {
    if factor == 1.0 {
        return;
    }
    for value in slice.iter_mut() {
        *value = (*value - mean) * factor + mean;
    }
}

fn apply_saturation(r: &mut f32, g: &mut f32, b: &mut f32, factor: f32) {
    if factor == 1.0 {
        return;
    }
    let gray = 0.298_995_97 * *r + 0.587_096 * *g + 0.113_907_03 * *b;
    *r = (*r - gray) * factor + gray;
    *g = (*g - gray) * factor + gray;
    *b = (*b - gray) * factor + gray;
}

fn apply_hue(r: &mut f32, g: &mut f32, b: &mut f32, radians: f32) {
    if radians == 0.0 {
        return;
    }
    let cos_h = radians.cos();
    let sin_h = radians.sin();
    let y = 0.299 * *r + 0.587 * *g + 0.114 * *b;
    let u = -0.147_13 * *r - 0.288_86 * *g + 0.436 * *b;
    let v = 0.615 * *r - 0.514_99 * *g - 0.100_01 * *b;
    let u_prime = u * cos_h - v * sin_h;
    let v_prime = u * sin_h + v * cos_h;
    *r = y + 1.13983 * v_prime;
    *g = y - 0.39465 * u_prime - 0.58060 * v_prime;
    *b = y + 2.03211 * u_prime;
}

fn cpu_color_jitter(input: &[f32], config: ColorJitterConfig, means: [f32; 4]) -> Vec<f32> {
    let mut output = input.to_vec();
    if config.brightness != 1.0 {
        for value in output.iter_mut() {
            *value *= config.brightness;
        }
    }
    if config.contrast != 1.0 {
        let plane = config.height * config.width;
        for c in 0..config.channels {
            let start = c * plane;
            let end = start + plane;
            apply_contrast(&mut output[start..end], means[c.min(3)], config.contrast);
        }
    }
    if config.channels >= 3 && (config.saturation != 1.0 || config.hue != 0.0) {
        let plane = config.height * config.width;
        for idx in 0..plane {
            let mut r = output[idx];
            let mut g = output[plane + idx];
            let mut b = output[2 * plane + idx];
            if config.saturation != 1.0 {
                apply_saturation(&mut r, &mut g, &mut b, config.saturation);
            }
            if config.hue != 0.0 {
                apply_hue(&mut r, &mut g, &mut b, config.hue);
            }
            output[idx] = r;
            output[plane + idx] = g;
            output[2 * plane + idx] = b;
        }
    }
    output
}

fn gpu_resize(
    ctx: &GpuContext,
    input: &[f32],
    config: ResizeConfig,
) -> Result<Vec<f32>, TransformDispatchError> {
    ensure_blocking_readback_supported("host-visible resize")?;
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let in_buffer = upload_slice(
        device,
        "st.backend.transform.resize.input",
        input,
        BufferUsages::STORAGE,
    )?;
    let out_elements = TransformDispatcher::validate_volume(
        "resize destination",
        config.channels,
        config.dst_height,
        config.dst_width,
    )?;
    let out_buffer = dispatch_resize_buffer(ctx, &in_buffer, config)?;
    read_buffer(
        device,
        queue,
        &out_buffer,
        out_elements,
        "st.backend.transform.resize.readback",
    )
    .map_err(TransformDispatchError::from)
}

fn gpu_center_crop(
    ctx: &GpuContext,
    input: &[f32],
    config: CenterCropConfig,
) -> Result<Vec<f32>, TransformDispatchError> {
    ensure_blocking_readback_supported("host-visible center crop")?;
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let in_buffer = upload_slice(
        device,
        "st.backend.transform.crop.input",
        input,
        BufferUsages::STORAGE,
    )?;
    let out_elements = TransformDispatcher::validate_volume(
        "center crop destination",
        config.channels,
        config.crop_height,
        config.crop_width,
    )?;
    let out_buffer = dispatch_center_crop_buffer(ctx, &in_buffer, config)?;
    read_buffer(
        device,
        queue,
        &out_buffer,
        out_elements,
        "st.backend.transform.crop.readback",
    )
    .map_err(TransformDispatchError::from)
}

fn gpu_horizontal_flip(
    ctx: &GpuContext,
    input: &[f32],
    config: HorizontalFlipConfig,
) -> Result<Vec<f32>, TransformDispatchError> {
    ensure_blocking_readback_supported("host-visible horizontal flip")?;
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let in_buffer = upload_slice(
        device,
        "st.backend.transform.flip.input",
        input,
        BufferUsages::STORAGE,
    )?;
    let out_elements = TransformDispatcher::validate_volume(
        "horizontal flip",
        config.channels,
        config.height,
        config.width,
    )?;
    let out_buffer = dispatch_horizontal_flip_buffer(ctx, &in_buffer, config)?;
    read_buffer(
        device,
        queue,
        &out_buffer,
        out_elements,
        "st.backend.transform.flip.readback",
    )
    .map_err(TransformDispatchError::from)
}

fn gpu_color_jitter(
    ctx: &GpuContext,
    input: &[f32],
    config: ColorJitterConfig,
    means: [f32; 4],
) -> Result<Vec<f32>, TransformDispatchError> {
    ensure_blocking_readback_supported("host-visible color jitter")?;
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let in_buffer = upload_slice(
        device,
        "st.backend.transform.jitter.input",
        input,
        BufferUsages::STORAGE,
    )?;
    let out_elements = TransformDispatcher::validate_volume(
        "color jitter",
        config.channels,
        config.height,
        config.width,
    )?;
    let out_buffer = empty_buffer::<f32>(
        device,
        "st.backend.transform.jitter.output",
        out_elements,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    )?;
    let params = ColorJitterParams {
        dims: [
            shader_dimension("height", config.height)?,
            shader_dimension("width", config.width)?,
            shader_dimension("channels", config.channels)?,
            0,
        ],
        factors: [
            config.brightness,
            config.contrast,
            config.saturation,
            config.hue,
        ],
        means,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.backend.transform.jitter.params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let bind_group = ctx.bind_group(&in_buffer, &out_buffer, &params_buffer);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.backend.transform.jitter.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.backend.transform.jitter.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ctx.pipelines.color_jitter.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy, gz) = workgroup_dims(
            device,
            config.width,
            config.height,
            config.channels,
            16,
            16,
            1,
        )?;
        pass.dispatch_workgroups(gx, gy, gz);
    }
    queue.submit(std::iter::once(encoder.finish()));
    read_buffer(
        device,
        queue,
        &out_buffer,
        out_elements,
        "st.backend.transform.jitter.readback",
    )
    .map_err(TransformDispatchError::from)
}

fn dispatch_resize_buffer(
    ctx: &GpuContext,
    input: &Buffer,
    config: ResizeConfig,
) -> Result<Buffer, TransformDispatchError> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let out_elements = TransformDispatcher::validate_volume(
        "resize destination",
        config.channels,
        config.dst_height,
        config.dst_width,
    )?;
    let out_buffer = empty_buffer::<f32>(
        device,
        "st.backend.transform.resize.seq.output",
        out_elements,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    )?;
    let params = ResizeParams {
        src_height: shader_dimension("source height", config.src_height)?,
        src_width: shader_dimension("source width", config.src_width)?,
        dst_height: shader_dimension("destination height", config.dst_height)?,
        dst_width: shader_dimension("destination width", config.dst_width)?,
        channels: shader_dimension("channels", config.channels)?,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.backend.transform.resize.seq.params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let bind_group = ctx.bind_group(input, &out_buffer, &params_buffer);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.backend.transform.resize.seq.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.backend.transform.resize.seq.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ctx.pipelines.resize.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy, gz) = workgroup_dims(
            device,
            config.dst_width,
            config.dst_height,
            config.channels,
            8,
            8,
            1,
        )?;
        pass.dispatch_workgroups(gx, gy, gz);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(out_buffer)
}

fn dispatch_center_crop_buffer(
    ctx: &GpuContext,
    input: &Buffer,
    config: CenterCropConfig,
) -> Result<Buffer, TransformDispatchError> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let out_elements = TransformDispatcher::validate_volume(
        "center crop destination",
        config.channels,
        config.crop_height,
        config.crop_width,
    )?;
    let out_buffer = empty_buffer::<f32>(
        device,
        "st.backend.transform.crop.seq.output",
        out_elements,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    )?;
    let params = CropParams {
        src_height: shader_dimension("source height", config.src_height)?,
        src_width: shader_dimension("source width", config.src_width)?,
        dst_height: shader_dimension("crop height", config.crop_height)?,
        dst_width: shader_dimension("crop width", config.crop_width)?,
        top: shader_dimension("crop top", (config.src_height - config.crop_height) / 2)?,
        left: shader_dimension("crop left", (config.src_width - config.crop_width) / 2)?,
        channels: shader_dimension("channels", config.channels)?,
        _pad: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.backend.transform.crop.seq.params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let bind_group = ctx.bind_group(input, &out_buffer, &params_buffer);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.backend.transform.crop.seq.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.backend.transform.crop.seq.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ctx.pipelines.center_crop.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy, gz) = workgroup_dims(
            device,
            config.crop_width,
            config.crop_height,
            config.channels,
            8,
            8,
            1,
        )?;
        pass.dispatch_workgroups(gx, gy, gz);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(out_buffer)
}

fn dispatch_horizontal_flip_buffer(
    ctx: &GpuContext,
    input: &Buffer,
    config: HorizontalFlipConfig,
) -> Result<Buffer, TransformDispatchError> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let out_elements = TransformDispatcher::validate_volume(
        "horizontal flip",
        config.channels,
        config.height,
        config.width,
    )?;
    let out_buffer = empty_buffer::<f32>(
        device,
        "st.backend.transform.flip.seq.output",
        out_elements,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    )?;
    let params = FlipParams {
        height: shader_dimension("height", config.height)?,
        width: shader_dimension("width", config.width)?,
        channels: shader_dimension("channels", config.channels)?,
        apply: config.apply as u32,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.backend.transform.flip.seq.params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let bind_group = ctx.bind_group(input, &out_buffer, &params_buffer);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.backend.transform.flip.seq.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.backend.transform.flip.seq.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(ctx.pipelines.horizontal_flip.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy, gz) = workgroup_dims(
            device,
            config.width,
            config.height,
            config.channels,
            16,
            16,
            1,
        )?;
        pass.dispatch_workgroups(gx, gy, gz);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(out_buffer)
}

fn validate_geometry_transition(
    current: ImageGeometry,
    command: GeometryCommand,
) -> Result<ImageGeometry, TransformDispatchError> {
    match command {
        GeometryCommand::Resize(config) => {
            if config.channels != current.channels
                || config.src_height != current.height
                || config.src_width != current.width
            {
                return Err(TransformDispatchError::InvalidGeometry(
                    "resize config incompatible with input geometry".into(),
                ));
            }
            TransformDispatcher::validate_volume(
                "resize source",
                config.channels,
                config.src_height,
                config.src_width,
            )?;
            TransformDispatcher::validate_volume(
                "resize destination",
                config.channels,
                config.dst_height,
                config.dst_width,
            )?;
            Ok(ImageGeometry {
                channels: current.channels,
                height: config.dst_height,
                width: config.dst_width,
            })
        }
        GeometryCommand::CenterCrop(config) => {
            if config.channels != current.channels
                || config.src_height != current.height
                || config.src_width != current.width
            {
                return Err(TransformDispatchError::InvalidGeometry(
                    "center crop config incompatible with input geometry".into(),
                ));
            }
            TransformDispatcher::validate_volume(
                "center crop source",
                config.channels,
                config.src_height,
                config.src_width,
            )?;
            TransformDispatcher::validate_volume(
                "center crop destination",
                config.channels,
                config.crop_height,
                config.crop_width,
            )?;
            if config.crop_height > config.src_height || config.crop_width > config.src_width {
                return Err(TransformDispatchError::InvalidGeometry(
                    "crop must fit inside source".into(),
                ));
            }
            Ok(ImageGeometry {
                channels: current.channels,
                height: config.crop_height,
                width: config.crop_width,
            })
        }
        GeometryCommand::HorizontalFlip(config) => {
            if config.channels != current.channels
                || config.height != current.height
                || config.width != current.width
            {
                return Err(TransformDispatchError::InvalidGeometry(
                    "horizontal flip config incompatible with input geometry".into(),
                ));
            }
            TransformDispatcher::validate_volume(
                "horizontal flip",
                config.channels,
                config.height,
                config.width,
            )?;
            Ok(current)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_geometry_reports_overflow_instead_of_panicking() {
        let geometry = ImageGeometry {
            channels: usize::MAX,
            height: 2,
            width: 1,
        };
        assert!(matches!(
            geometry.element_count(),
            Err(TransformDispatchError::InvalidGeometry(_))
        ));
    }

    #[test]
    fn empty_sequence_still_validates_input_contract() {
        let dispatcher = TransformDispatcher::cpu();
        let result = dispatcher.run_geometry_sequence(
            &[],
            ImageGeometry {
                channels: 1,
                height: 1,
                width: 1,
            },
            &[],
        );
        assert!(matches!(
            result,
            Err(TransformDispatchError::InvalidGeometry(_))
        ));
    }

    #[test]
    fn color_jitter_gpu_matches_cpu_for_more_than_eight_channels_when_enabled() {
        if std::env::var_os("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").is_none() {
            return;
        }
        let gpu = match TransformDispatcher::new_default_gpu() {
            Ok(dispatcher) => dispatcher,
            Err(TransformDispatchError::Runtime(WgpuRuntimeError::NoAdapter)) => {
                eprintln!("skipping transform runtime test: no WGPU adapter");
                return;
            }
            Err(error) => panic!("failed to create transform dispatcher: {error}"),
        };
        let cpu = TransformDispatcher::cpu();
        let config = ColorJitterConfig {
            channels: 9,
            height: 2,
            width: 3,
            brightness: 1.1,
            contrast: 0.8,
            saturation: 0.7,
            hue: 0.17,
        };
        let input = (0..config.channels * config.height * config.width)
            .map(|index| (index as f32 - 17.0) / 13.0)
            .collect::<Vec<_>>();
        let expected = cpu.color_jitter(&input, config).unwrap();
        let actual = gpu.color_jitter(&input, config).unwrap();
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "color jitter mismatch at {index}: actual={actual} expected={expected}"
            );
        }
    }
}
