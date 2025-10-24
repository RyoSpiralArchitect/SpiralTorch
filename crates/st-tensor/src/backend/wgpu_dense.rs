// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "wgpu")]

use crate::util::readback_f32;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device, PipelineLayout, Queue};

const MATMUL_WGSL_TEMPLATE: &str = include_str!("../wgpu_shaders/dense_matmul.wgsl");
const FUSED_CONV_WGSL_TEMPLATE: &str = include_str!("../wgpu_shaders/fused_im2col_matmul.wgsl");
const FUSED_LINEAR_WGSL_TEMPLATE: &str =
    include_str!("../wgpu_shaders/fused_matmul_bias_relu.wgsl");
const FUSED_LINEAR_GELU_WGSL_TEMPLATE: &str =
    include_str!("../wgpu_shaders/fused_matmul_bias_gelu.wgsl");
const FUSED_LINEAR_RESIDUAL_WGSL_TEMPLATE: &str =
    include_str!("../wgpu_shaders/fused_matmul_bias_residual_relu.wgsl");
const FUSED_LINEAR_RESIDUAL_GELU_WGSL_TEMPLATE: &str =
    include_str!("../wgpu_shaders/fused_matmul_bias_residual_gelu.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulParams {
    rows: u32,
    cols: u32,
    inner: u32,
    _pad: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TileConfig {
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
}

impl TileConfig {
    const fn new(tile_m: u32, tile_n: u32, tile_k: u32) -> Self {
        Self {
            tile_m,
            tile_n,
            tile_k,
        }
    }

    const fn tile_m(&self) -> u32 {
        self.tile_m
    }

    const fn tile_n(&self) -> u32 {
        self.tile_n
    }

    const fn tile_k(&self) -> u32 {
        self.tile_k
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FusedActivation {
    Relu,
    Gelu,
}

struct DenseContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    bind_layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
    matmul_pipelines: Mutex<HashMap<TileConfig, Arc<ComputePipeline>>>,
    fused_linear_layout: BindGroupLayout,
    fused_linear_pipeline_layout: PipelineLayout,
    fused_linear_pipelines: Mutex<HashMap<(TileConfig, FusedActivation), Arc<ComputePipeline>>>,
    fused_linear_residual_layout: BindGroupLayout,
    fused_linear_residual_pipeline_layout: PipelineLayout,
    fused_linear_residual_pipelines:
        Mutex<HashMap<(TileConfig, FusedActivation), Arc<ComputePipeline>>>,
    fused_conv_layout: BindGroupLayout,
    fused_conv_pipeline_layout: PipelineLayout,
    fused_conv_pipelines: Mutex<HashMap<TileConfig, Arc<ComputePipeline>>>,
}

impl DenseContext {
    fn new() -> Result<Self, String> {
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
        .ok_or_else(|| "no suitable WGPU adapter".to_string())?;

        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        required_features: wgpu::Features::empty(),
                        required_limits: adapter.limits(),
                    },
                    None,
                )
                .await
        })
        .map_err(|err| err.to_string())?;

        let device: Arc<Device> = Arc::new(device);
        let queue: Arc<Queue> = Arc::new(queue);

        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("st.tensor.wgpu_dense.bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("st.tensor.wgpu_dense.pipeline_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let fused_linear_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_linear_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let fused_linear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_linear_pipeline_layout"),
                bind_group_layouts: &[&fused_linear_layout],
                push_constant_ranges: &[],
            });

        let fused_linear_residual_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_linear_residual_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let fused_linear_residual_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_linear_residual_pipeline_layout"),
                bind_group_layouts: &[&fused_linear_residual_layout],
                push_constant_ranges: &[],
            });

        let fused_conv_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_conv_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let fused_conv_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_conv_pipeline_layout"),
                bind_group_layouts: &[&fused_conv_layout],
                push_constant_ranges: &[],
            });

        Ok(Self {
            device,
            queue,
            bind_layout,
            pipeline_layout,
            matmul_pipelines: Mutex::new(HashMap::new()),
            fused_linear_layout,
            fused_linear_pipeline_layout,
            fused_linear_pipelines: Mutex::new(HashMap::new()),
            fused_linear_residual_layout,
            fused_linear_residual_pipeline_layout,
            fused_linear_residual_pipelines: Mutex::new(HashMap::new()),
            fused_conv_layout,
            fused_conv_pipeline_layout,
            fused_conv_pipelines: Mutex::new(HashMap::new()),
        })
    }

    fn device(&self) -> &Device {
        self.device.as_ref()
    }

    fn queue(&self) -> &Queue {
        self.queue.as_ref()
    }

    fn pipeline_for(&self, config: TileConfig) -> Arc<ComputePipeline> {
        let mut pipelines = self.matmul_pipelines.lock().unwrap();
        if let Some(pipeline) = pipelines.get(&config) {
            return pipeline.clone();
        }

        let shader_source = format!(
            MATMUL_WGSL_TEMPLATE,
            tile_m = config.tile_m(),
            tile_n = config.tile_n(),
            tile_k = config.tile_k(),
        );
        let shader_label = format!(
            "st.tensor.wgpu_dense.matmul_shader.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&shader_label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let pipeline_label = format!(
            "st.tensor.wgpu_dense.matmul_pipeline.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&pipeline_label),
                layout: Some(&self.pipeline_layout),
                module: &shader,
                entry_point: "main",
            },
        ));
        pipelines.insert(config, pipeline.clone());
        pipeline
    }

    fn fused_linear_pipeline_for(
        &self,
        config: TileConfig,
        activation: FusedActivation,
    ) -> Arc<ComputePipeline> {
        let mut pipelines = self.fused_linear_pipelines.lock().unwrap();
        let key = (config, activation);
        if let Some(pipeline) = pipelines.get(&key) {
            return pipeline.clone();
        }

        let (shader_template, activation_label) = match activation {
            FusedActivation::Relu => (FUSED_LINEAR_WGSL_TEMPLATE, "relu"),
            FusedActivation::Gelu => (FUSED_LINEAR_GELU_WGSL_TEMPLATE, "gelu"),
        };
        let shader_source = format!(
            shader_template,
            tile_m = config.tile_m(),
            tile_n = config.tile_n(),
            tile_k = config.tile_k(),
        );
        let shader_label = format!(
            "st.tensor.wgpu_dense.fused_linear_shader.{activation_label}.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&shader_label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let pipeline_label = format!(
            "st.tensor.wgpu_dense.fused_linear_pipeline.{activation_label}.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&pipeline_label),
                layout: Some(&self.fused_linear_pipeline_layout),
                module: &shader,
                entry_point: "main",
            },
        ));
        pipelines.insert(key, pipeline.clone());
        pipeline
    }

    fn fused_linear_with_residual_pipeline_for(
        &self,
        config: TileConfig,
        activation: FusedActivation,
    ) -> Arc<ComputePipeline> {
        let mut pipelines = self.fused_linear_residual_pipelines.lock().unwrap();
        let key = (config, activation);
        if let Some(pipeline) = pipelines.get(&key) {
            return pipeline.clone();
        }

        let (shader_template, activation_label) = match activation {
            FusedActivation::Relu => (FUSED_LINEAR_RESIDUAL_WGSL_TEMPLATE, "relu"),
            FusedActivation::Gelu => (FUSED_LINEAR_RESIDUAL_GELU_WGSL_TEMPLATE, "gelu"),
        };
        let shader_source = format!(
            shader_template,
            tile_m = config.tile_m(),
            tile_n = config.tile_n(),
            tile_k = config.tile_k(),
        );
        let shader_label = format!(
            "st.tensor.wgpu_dense.fused_linear_residual_shader.{activation_label}.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&shader_label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let pipeline_label = format!(
            "st.tensor.wgpu_dense.fused_linear_residual_pipeline.{activation_label}.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&pipeline_label),
                layout: Some(&self.fused_linear_residual_pipeline_layout),
                module: &shader,
                entry_point: "main",
            },
        ));
        pipelines.insert(key, pipeline.clone());
        pipeline
    }

    fn fused_conv_pipeline_for(&self, config: TileConfig) -> Arc<ComputePipeline> {
        let mut pipelines = self.fused_conv_pipelines.lock().unwrap();
        if let Some(pipeline) = pipelines.get(&config) {
            return pipeline.clone();
        }

        let shader_source = format!(
            FUSED_CONV_WGSL_TEMPLATE,
            tile_m = config.tile_m(),
            tile_n = config.tile_n(),
            tile_k = config.tile_k(),
        );
        let shader_label = format!(
            "st.tensor.wgpu_dense.fused_conv_shader.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&shader_label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
        let pipeline_label = format!(
            "st.tensor.wgpu_dense.fused_conv_pipeline.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(&pipeline_label),
                layout: Some(&self.fused_conv_pipeline_layout),
                module: &shader,
                entry_point: "main",
            },
        ));
        pipelines.insert(config, pipeline.clone());
        pipeline
    }

    fn bind_group(&self, a: &Buffer, b: &Buffer, c: &Buffer, params: &Buffer) -> BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.bind_group"),
            layout: &self.bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    fn fused_linear_bind_group(
        &self,
        lhs: &Buffer,
        rhs: &Buffer,
        bias: &Buffer,
        out: &Buffer,
        params: &Buffer,
    ) -> BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_linear_bind_group"),
            layout: &self.fused_linear_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    fn fused_linear_with_residual_bind_group(
        &self,
        lhs: &Buffer,
        rhs: &Buffer,
        bias: &Buffer,
        residual: &Buffer,
        out: &Buffer,
        params: &Buffer,
    ) -> BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_linear_residual_bind_group"),
            layout: &self.fused_linear_residual_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: residual.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    fn fused_conv_bind_group(
        &self,
        input: &Buffer,
        weights: &Buffer,
        output: &Buffer,
        params: &Buffer,
    ) -> BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_conv_bind_group"),
            layout: &self.fused_conv_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weights.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }
}

static CONTEXT: OnceLock<Arc<DenseContext>> = OnceLock::new();

fn dense_context() -> Result<Arc<DenseContext>, String> {
    if let Some(ctx) = CONTEXT.get() {
        return Ok(ctx.clone());
    }
    let ctx = Arc::new(DenseContext::new()?);
    let _ = CONTEXT.set(ctx.clone());
    Ok(ctx)
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvGemmParams {
    batch: u32,
    in_channels: u32,
    input_h: u32,
    input_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: i32,
    pad_w: i32,
    dilation_h: u32,
    dilation_w: u32,
    out_h: u32,
    out_w: u32,
    span: u32,
    out_channels: u32,
    _pad0: u32,
    _pad1: u32,
}

pub fn matmul(
    a: &[f32],
    b: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    if rows == 0 || inner == 0 || cols == 0 {
        return Err("matrix dimensions must be positive".into());
    }
    if a.len() != rows * inner {
        return Err(format!(
            "lhs buffer length mismatch: expected {} elements, got {}",
            rows * inner,
            a.len()
        ));
    }
    if b.len() != inner * cols {
        return Err(format!(
            "rhs buffer length mismatch: expected {} elements, got {}",
            inner * cols,
            b.len()
        ));
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.a"),
        contents: bytemuck::cast_slice(a),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.b"),
        contents: bytemuck::cast_slice(b),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_size = (rows * cols * std::mem::size_of::<f32>()) as u64;
    let c_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.tensor.wgpu_dense.c"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.encoder"),
    });
    let tile_config = select_tile_config(rows, inner, cols);
    dispatch_matmul(
        &ctx,
        &mut encoder,
        &a_buf,
        &b_buf,
        &c_buf,
        rows,
        inner,
        cols,
        tile_config,
    );
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &c_buf, rows * cols)
}

pub fn matmul_bias_relu(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    if rows == 0 || inner == 0 || cols == 0 {
        return Err("matrix dimensions must be positive".into());
    }
    if lhs.len() != rows * inner {
        return Err(format!(
            "lhs buffer length mismatch: expected {} elements, got {}",
            rows * inner,
            lhs.len()
        ));
    }
    if rhs.len() != inner * cols {
        return Err(format!(
            "rhs buffer length mismatch: expected {} elements, got {}",
            inner * cols,
            rhs.len()
        ));
    }
    if bias.len() != cols {
        return Err(format!(
            "bias length mismatch: expected {} elements, got {}",
            cols,
            bias.len()
        ));
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let lhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear.lhs"),
        contents: bytemuck::cast_slice(lhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let rhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear.rhs"),
        contents: bytemuck::cast_slice(rhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear.bias"),
        contents: bytemuck::cast_slice(bias),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_size = (rows * cols * std::mem::size_of::<f32>()) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear.out"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear.encoder"),
    });
    let tile_config = select_tile_config(rows, inner, cols);
    dispatch_fused_linear(
        &ctx,
        &mut encoder,
        &lhs_buf,
        &rhs_buf,
        &bias_buf,
        &out_buf,
        rows,
        inner,
        cols,
        tile_config,
        FusedActivation::Relu,
    );
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn matmul_bias_gelu(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    if rows == 0 || inner == 0 || cols == 0 {
        return Err("matrix dimensions must be positive".into());
    }
    if lhs.len() != rows * inner {
        return Err(format!(
            "lhs buffer length mismatch: expected {} elements, got {}",
            rows * inner,
            lhs.len()
        ));
    }
    if rhs.len() != inner * cols {
        return Err(format!(
            "rhs buffer length mismatch: expected {} elements, got {}",
            inner * cols,
            rhs.len()
        ));
    }
    if bias.len() != cols {
        return Err(format!(
            "bias length mismatch: expected {} elements, got {}",
            cols,
            bias.len()
        ));
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let lhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_gelu.lhs"),
        contents: bytemuck::cast_slice(lhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let rhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_gelu.rhs"),
        contents: bytemuck::cast_slice(rhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_gelu.bias"),
        contents: bytemuck::cast_slice(bias),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_size = (rows * cols * std::mem::size_of::<f32>()) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_gelu.out"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_gelu.encoder"),
    });
    let tile_config = select_tile_config(rows, inner, cols);
    dispatch_fused_linear(
        &ctx,
        &mut encoder,
        &lhs_buf,
        &rhs_buf,
        &bias_buf,
        &out_buf,
        rows,
        inner,
        cols,
        tile_config,
        FusedActivation::Gelu,
    );
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn matmul_bias_add_relu(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    residual: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    if rows == 0 || inner == 0 || cols == 0 {
        return Err("matrix dimensions must be positive".into());
    }
    if lhs.len() != rows * inner {
        return Err(format!(
            "lhs buffer length mismatch: expected {} elements, got {}",
            rows * inner,
            lhs.len()
        ));
    }
    if rhs.len() != inner * cols {
        return Err(format!(
            "rhs buffer length mismatch: expected {} elements, got {}",
            inner * cols,
            rhs.len()
        ));
    }
    if bias.len() != cols {
        return Err(format!(
            "bias length mismatch: expected {} elements, got {}",
            cols,
            bias.len()
        ));
    }
    if residual.len() != rows * cols {
        return Err(format!(
            "residual length mismatch: expected {} elements, got {}",
            rows * cols,
            residual.len()
        ));
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let lhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual.lhs"),
        contents: bytemuck::cast_slice(lhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let rhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual.rhs"),
        contents: bytemuck::cast_slice(rhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual.bias"),
        contents: bytemuck::cast_slice(bias),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let residual_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual.residual"),
        contents: bytemuck::cast_slice(residual),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_size = (rows * cols * std::mem::size_of::<f32>()) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual.out"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual.encoder"),
    });
    let tile_config = select_tile_config(rows, inner, cols);
    dispatch_fused_linear_with_residual(
        &ctx,
        &mut encoder,
        &lhs_buf,
        &rhs_buf,
        &bias_buf,
        &residual_buf,
        &out_buf,
        rows,
        inner,
        cols,
        tile_config,
        FusedActivation::Relu,
    );
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn matmul_bias_add_gelu(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    residual: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    if rows == 0 || inner == 0 || cols == 0 {
        return Err("matrix dimensions must be positive".into());
    }
    if lhs.len() != rows * inner {
        return Err(format!(
            "lhs buffer length mismatch: expected {} elements, got {}",
            rows * inner,
            lhs.len()
        ));
    }
    if rhs.len() != inner * cols {
        return Err(format!(
            "rhs buffer length mismatch: expected {} elements, got {}",
            inner * cols,
            rhs.len()
        ));
    }
    if bias.len() != cols {
        return Err(format!(
            "bias length mismatch: expected {} elements, got {}",
            cols,
            bias.len()
        ));
    }
    if residual.len() != rows * cols {
        return Err(format!(
            "residual length mismatch: expected {} elements, got {}",
            rows * cols,
            residual.len()
        ));
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let lhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual_gelu.lhs"),
        contents: bytemuck::cast_slice(lhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let rhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual_gelu.rhs"),
        contents: bytemuck::cast_slice(rhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual_gelu.bias"),
        contents: bytemuck::cast_slice(bias),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let residual_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual_gelu.residual"),
        contents: bytemuck::cast_slice(residual),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let result_size = (rows * cols * std::mem::size_of::<f32>()) as u64;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual_gelu.out"),
        size: result_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.fused_linear_residual_gelu.encoder"),
    });
    let tile_config = select_tile_config(rows, inner, cols);
    dispatch_fused_linear_with_residual(
        &ctx,
        &mut encoder,
        &lhs_buf,
        &rhs_buf,
        &bias_buf,
        &residual_buf,
        &out_buf,
        rows,
        inner,
        cols,
        tile_config,
        FusedActivation::Gelu,
    );
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn is_available() -> bool {
    dense_context().is_ok()
}

fn dispatch_matmul(
    ctx: &DenseContext,
    encoder: &mut wgpu::CommandEncoder,
    lhs: &Buffer,
    rhs: &Buffer,
    out: &Buffer,
    rows: usize,
    inner: usize,
    cols: usize,
    tile: TileConfig,
) {
    let params = MatmulParams {
        rows: rows as u32,
        cols: cols as u32,
        inner: inner as u32,
        _pad: 0,
    };
    let params_buf = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.tensor.wgpu_dense.matmul_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind_group = ctx.bind_group(lhs, rhs, out, &params_buf);
    let pipeline = ctx.pipeline_for(tile);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.matmul_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        let groups_x = ((cols as u32) + tile.tile_n() - 1) / tile.tile_n();
        let groups_y = ((rows as u32) + tile.tile_m() - 1) / tile.tile_m();
        pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), 1);
    }
}

fn dispatch_fused_linear(
    ctx: &DenseContext,
    encoder: &mut wgpu::CommandEncoder,
    lhs: &Buffer,
    rhs: &Buffer,
    bias: &Buffer,
    out: &Buffer,
    rows: usize,
    inner: usize,
    cols: usize,
    tile: TileConfig,
    activation: FusedActivation,
) {
    let params = MatmulParams {
        rows: rows as u32,
        cols: cols as u32,
        inner: inner as u32,
        _pad: 0,
    };
    let params_buf = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_linear.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind_group = ctx.fused_linear_bind_group(lhs, rhs, bias, out, &params_buf);
    let pipeline = ctx.fused_linear_pipeline_for(tile, activation);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_linear.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        let groups_x = ((cols as u32) + tile.tile_n() - 1) / tile.tile_n();
        let groups_y = ((rows as u32) + tile.tile_m() - 1) / tile.tile_m();
        pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), 1);
    }
}

fn dispatch_fused_linear_with_residual(
    ctx: &DenseContext,
    encoder: &mut wgpu::CommandEncoder,
    lhs: &Buffer,
    rhs: &Buffer,
    bias: &Buffer,
    residual: &Buffer,
    out: &Buffer,
    rows: usize,
    inner: usize,
    cols: usize,
    tile: TileConfig,
    activation: FusedActivation,
) {
    let params = MatmulParams {
        rows: rows as u32,
        cols: cols as u32,
        inner: inner as u32,
        _pad: 0,
    };
    let params_buf = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_linear_residual.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind_group =
        ctx.fused_linear_with_residual_bind_group(lhs, rhs, bias, residual, out, &params_buf);
    let pipeline = ctx.fused_linear_with_residual_pipeline_for(tile, activation);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_linear_residual.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        let groups_x = ((cols as u32) + tile.tile_n() - 1) / tile.tile_n();
        let groups_y = ((rows as u32) + tile.tile_m() - 1) / tile.tile_m();
        pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), 1);
    }
}

pub fn conv_im2col_gemm(
    input: &[f32],
    batch: usize,
    in_channels: usize,
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: i32,
    pad_w: i32,
    dilation_h: usize,
    dilation_w: usize,
    weight_t: &[f32],
    out_channels: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<f32>, String> {
    if batch == 0
        || in_channels == 0
        || kernel_h == 0
        || kernel_w == 0
        || out_channels == 0
        || out_h == 0
        || out_w == 0
    {
        return Err("convolution dimensions must be positive".into());
    }
    let span = in_channels
        .checked_mul(kernel_h)
        .and_then(|value| value.checked_mul(kernel_w))
        .ok_or_else(|| "kernel span overflow".to_string())?;
    let rows = batch
        .checked_mul(out_h)
        .and_then(|value| value.checked_mul(out_w))
        .ok_or_else(|| "output spatial overflow".to_string())?;
    if input.len() != batch * in_channels * input_h * input_w {
        return Err("input buffer length mismatch".into());
    }
    if weight_t.len() != span * out_channels {
        return Err("transposed weight buffer length mismatch".into());
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.conv.input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let conv_params = ConvGemmParams {
        batch: batch as u32,
        in_channels: in_channels as u32,
        input_h: input_h as u32,
        input_w: input_w as u32,
        kernel_h: kernel_h as u32,
        kernel_w: kernel_w as u32,
        stride_h: stride_h as u32,
        stride_w: stride_w as u32,
        pad_h,
        pad_w,
        dilation_h: dilation_h as u32,
        dilation_w: dilation_w as u32,
        out_h: out_h as u32,
        out_w: out_w as u32,
        span: span as u32,
        out_channels: out_channels as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let conv_params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.conv.params"),
        contents: bytemuck::bytes_of(&conv_params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let weight_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.conv.weight_t"),
        contents: bytemuck::cast_slice(weight_t),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_size = (rows * out_channels * std::mem::size_of::<f32>()) as u64;
    let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("st.tensor.wgpu_dense.conv.output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.conv.encoder"),
    });
    let tile_config = select_tile_config(rows, span, out_channels);
    let fused_pipeline = ctx.fused_conv_pipeline_for(tile_config);
    let fused_bind_group =
        ctx.fused_conv_bind_group(&input_buf, &weight_buf, &output_buf, &conv_params_buf);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.conv.fused_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(fused_pipeline.as_ref());
        pass.set_bind_group(0, &fused_bind_group, &[]);
        let groups_x = ((out_channels as u32) + tile_config.tile_n() - 1) / tile_config.tile_n();
        let groups_y = ((rows as u32) + tile_config.tile_m() - 1) / tile_config.tile_m();
        pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), 1);
    }

    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &output_buf, rows * out_channels)
}

fn select_tile_config(rows: usize, inner: usize, cols: usize) -> TileConfig {
    let rows = rows as u32;
    let cols = cols as u32;
    let inner = inner as u32;

    if rows <= 32 && cols <= 32 {
        return TileConfig::new(8, 8, 8);
    }

    if inner <= 64 {
        if rows > cols.saturating_mul(2) {
            return TileConfig::new(32, 8, 8);
        }
        if cols > rows.saturating_mul(2) {
            return TileConfig::new(8, 32, 8);
        }
        return TileConfig::new(16, 16, 8);
    }

    if rows > cols.saturating_mul(2) {
        return TileConfig::new(32, 8, 16);
    }

    if cols > rows.saturating_mul(2) {
        return TileConfig::new(8, 32, 16);
    }

    TileConfig::new(16, 16, 16)
}

pub fn should_use(rows: usize, inner: usize, cols: usize) -> bool {
    if rows == 0 || inner == 0 || cols == 0 {
        return false;
    }

    let volume = rows.saturating_mul(inner).saturating_mul(cols);

    volume >= 32 * 32 * 32
}
