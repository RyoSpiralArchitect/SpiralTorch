// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::{ShaderCache, ShaderLoadError};
use bytemuck::{Pod, Zeroable};
use st_tensor::backend::wgpu_util::{empty_buffer, read_buffer, upload_slice, WgpuContext};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;
use wgpu::util::DeviceExt;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferUsages, ComputePipeline, Device,
    PipelineLayoutDescriptor, Queue, ShaderStages,
};

const TRANSFORM_SHADER_DIR: &str = "shaders/transforms";

#[derive(Debug, Error)]
pub enum TransformDispatchError {
    #[error("wgpu adapter unavailable")]
    NoAdapter,
    #[error("wgpu device request failed: {0}")]
    Device(String),
    #[error(transparent)]
    Shader(#[from] ShaderLoadError),
    #[error("invalid transform geometry: {0}")]
    InvalidGeometry(String),
    #[error("wgpu buffer readback failed: {0}")]
    Readback(String),
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
    resize: ComputePipeline,
    center_crop: ComputePipeline,
    horizontal_flip: ComputePipeline,
    color_jitter: ComputePipeline,
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
        let mut cache = ShaderCache::new(shader_root);
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
        device: Arc<Device>,
        queue: Arc<Queue>,
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

    pub fn new_default_gpu() -> Result<Self, TransformDispatchError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
        })
        .ok_or(TransformDispatchError::NoAdapter)?;

        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("st.backend.transform.device"),
                        required_features: wgpu::Features::empty(),
                        required_limits: adapter.limits(),
                    },
                    None,
                )
                .await
        })
        .map_err(|err| TransformDispatchError::Device(err.to_string()))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TRANSFORM_SHADER_DIR);
        Self::with_gpu(device, queue, shader_dir)
    }

    fn ensure_geometry(
        src_h: usize,
        src_w: usize,
        dst_h: usize,
        dst_w: usize,
    ) -> Result<(), TransformDispatchError> {
        if src_h == 0 || src_w == 0 || dst_h == 0 || dst_w == 0 {
            return Err(TransformDispatchError::InvalidGeometry(
                "spatial dimensions must be positive".into(),
            ));
        }
        Ok(())
    }

    pub fn resize(
        &self,
        input: &[f32],
        config: ResizeConfig,
    ) -> Result<Vec<f32>, TransformDispatchError> {
        Self::ensure_geometry(
            config.src_height,
            config.src_width,
            config.dst_height,
            config.dst_width,
        )?;
        let channels = config.channels;
        let expected = channels
            .checked_mul(config.src_height)
            .and_then(|v| v.checked_mul(config.src_width))
            .ok_or_else(|| {
                TransformDispatchError::InvalidGeometry("source volume overflow".into())
            })?;
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
        Self::ensure_geometry(
            config.src_height,
            config.src_width,
            config.crop_height,
            config.crop_width,
        )?;
        if config.crop_height > config.src_height || config.crop_width > config.src_width {
            return Err(TransformDispatchError::InvalidGeometry(
                "crop must fit inside source".into(),
            ));
        }
        let expected = config
            .channels
            .checked_mul(config.src_height)
            .and_then(|v| v.checked_mul(config.src_width))
            .ok_or_else(|| {
                TransformDispatchError::InvalidGeometry("source volume overflow".into())
            })?;
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
        Self::ensure_geometry(config.height, config.width, config.height, config.width)?;
        let expected = config
            .channels
            .checked_mul(config.height)
            .and_then(|v| v.checked_mul(config.width))
            .ok_or_else(|| {
                TransformDispatchError::InvalidGeometry("source volume overflow".into())
            })?;
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
        Self::ensure_geometry(config.height, config.width, config.height, config.width)?;
        let expected = config
            .channels
            .checked_mul(config.height)
            .and_then(|v| v.checked_mul(config.width))
            .ok_or_else(|| {
                TransformDispatchError::InvalidGeometry("source volume overflow".into())
            })?;
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
            for c in 0..config.channels.min(4) {
                let start = c * pixels;
                let end = start + pixels;
                let slice = &input[start..end];
                means[c] = slice.iter().sum::<f32>() / pixels as f32;
            }
        }
        match &self.backend {
            Backend::Cpu => Ok(cpu_color_jitter(input, config, means)),
            Backend::Gpu(ctx) => gpu_color_jitter(ctx, input, config, means),
        }
    }
}

fn workgroup_dims(
    width: usize,
    height: usize,
    depth: usize,
    x: u32,
    y: u32,
    z: u32,
) -> (u32, u32, u32) {
    let gx = ((width as u32) + x - 1) / x;
    let gy = ((height as u32) + y - 1) / y;
    let gz = ((depth as u32) + z - 1) / z;
    (gx, gy, gz)
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
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let in_buffer = upload_slice(
        device,
        "st.backend.transform.resize.input",
        input,
        BufferUsages::STORAGE,
    );
    let out_elements = config.channels * config.dst_height * config.dst_width;
    let out_buffer = empty_buffer(
        device,
        "st.backend.transform.resize.output",
        out_elements,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let params = ResizeParams {
        src_height: config.src_height as u32,
        src_width: config.src_width as u32,
        dst_height: config.dst_height as u32,
        dst_width: config.dst_width as u32,
        channels: config.channels as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.backend.transform.resize.params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let bind_group = ctx.bind_group(&in_buffer, &out_buffer, &params_buffer);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.backend.transform.resize.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.backend.transform.resize.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipelines.resize);
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy, gz) = workgroup_dims(
            config.dst_width,
            config.dst_height,
            config.channels,
            8,
            8,
            1,
        );
        pass.dispatch_workgroups(gx, gy, gz);
    }
    queue.submit(std::iter::once(encoder.finish()));
    read_buffer(device, queue, &out_buffer, out_elements).map_err(TransformDispatchError::Readback)
}

fn gpu_center_crop(
    ctx: &GpuContext,
    input: &[f32],
    config: CenterCropConfig,
) -> Result<Vec<f32>, TransformDispatchError> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let in_buffer = upload_slice(
        device,
        "st.backend.transform.crop.input",
        input,
        BufferUsages::STORAGE,
    );
    let out_elements = config.channels * config.crop_height * config.crop_width;
    let out_buffer = empty_buffer(
        device,
        "st.backend.transform.crop.output",
        out_elements,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let params = CropParams {
        src_height: config.src_height as u32,
        src_width: config.src_width as u32,
        dst_height: config.crop_height as u32,
        dst_width: config.crop_width as u32,
        top: ((config.src_height - config.crop_height) / 2) as u32,
        left: ((config.src_width - config.crop_width) / 2) as u32,
        channels: config.channels as u32,
        _pad: 0,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.backend.transform.crop.params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let bind_group = ctx.bind_group(&in_buffer, &out_buffer, &params_buffer);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.backend.transform.crop.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.backend.transform.crop.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipelines.center_crop);
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy, gz) = workgroup_dims(
            config.crop_width,
            config.crop_height,
            config.channels,
            8,
            8,
            1,
        );
        pass.dispatch_workgroups(gx, gy, gz);
    }
    queue.submit(std::iter::once(encoder.finish()));
    read_buffer(device, queue, &out_buffer, out_elements).map_err(TransformDispatchError::Readback)
}

fn gpu_horizontal_flip(
    ctx: &GpuContext,
    input: &[f32],
    config: HorizontalFlipConfig,
) -> Result<Vec<f32>, TransformDispatchError> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let in_buffer = upload_slice(
        device,
        "st.backend.transform.flip.input",
        input,
        BufferUsages::STORAGE,
    );
    let out_elements = config.channels * config.height * config.width;
    let out_buffer = empty_buffer(
        device,
        "st.backend.transform.flip.output",
        out_elements,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let params = FlipParams {
        height: config.height as u32,
        width: config.width as u32,
        channels: config.channels as u32,
        apply: config.apply as u32,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.backend.transform.flip.params"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    let bind_group = ctx.bind_group(&in_buffer, &out_buffer, &params_buffer);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.backend.transform.flip.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.backend.transform.flip.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.pipelines.horizontal_flip);
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy, gz) = workgroup_dims(config.width, config.height, config.channels, 16, 16, 1);
        pass.dispatch_workgroups(gx, gy, gz);
    }
    queue.submit(std::iter::once(encoder.finish()));
    read_buffer(device, queue, &out_buffer, out_elements).map_err(TransformDispatchError::Readback)
}

fn gpu_color_jitter(
    ctx: &GpuContext,
    input: &[f32],
    config: ColorJitterConfig,
    means: [f32; 4],
) -> Result<Vec<f32>, TransformDispatchError> {
    let device = ctx.context.device();
    let queue = ctx.context.queue();
    let in_buffer = upload_slice(
        device,
        "st.backend.transform.jitter.input",
        input,
        BufferUsages::STORAGE,
    );
    let out_elements = config.channels * config.height * config.width;
    let out_buffer = empty_buffer(
        device,
        "st.backend.transform.jitter.output",
        out_elements,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let params = ColorJitterParams {
        dims: [
            config.height as u32,
            config.width as u32,
            config.channels as u32,
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
        pass.set_pipeline(&ctx.pipelines.color_jitter);
        pass.set_bind_group(0, &bind_group, &[]);
        let (gx, gy, gz) = workgroup_dims(config.width, config.height, config.channels, 16, 16, 1);
        pass.dispatch_workgroups(gx, gy, gz);
    }
    queue.submit(std::iter::once(encoder.finish()));
    read_buffer(device, queue, &out_buffer, out_elements).map_err(TransformDispatchError::Readback)
}
