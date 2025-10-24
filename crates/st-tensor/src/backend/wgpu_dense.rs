// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "wgpu")]

use crate::backend::wgpu_util::WgpuContext;
use crate::pure::{PackedB, PackedLayout};
use crate::util::readback_f32;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use wgpu::util::DeviceExt;
use wgpu::{BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device, PipelineLayout, Queue};

const MATMUL_WGSL_TEMPLATE: &str = include_str!("../wgpu_shaders/dense_matmul.wgsl");
const FUSED_CONV_WGSL_TEMPLATE: &str = include_str!("../wgpu_shaders/fused_im2col_matmul.wgsl");
const ROW_SOFTMAX_WGSL: &str =
    include_str!("../../../st-backend-wgpu/src/shaders/softmax_workgroup.wgsl");
const FUSED_ATTENTION_WGSL_TEMPLATE: &str =
    include_str!("../../../st-backend-wgpu/src/shaders/fused_attention_online.wgsl");

const FUSED_ATTENTION_WORKGROUP: u32 = 128;
const FUSED_ATTENTION_MAX_HEAD_DIM: u32 = 256;

const FLAG_USE_BIAS: u32 = 1 << 0;
const FLAG_FUSED_RELU: u32 = 1 << 1;
const FLAG_FUSED_GELU: u32 = 1 << 2;
const FLAG_FUSED_RESIDUAL: u32 = 1 << 3;
const FLAG_USE_INT8: u32 = 1 << 4;
const FLAG_USE_F16: u32 = 1 << 5;

const QUANTIZATION_MIN_VOLUME: usize = 64 * 64;

const FUSED_ATTENTION_FLAG_USE_Z_BIAS: u32 = 1 << 0;
const FUSED_ATTENTION_FLAG_USE_ATTN_BIAS: u32 = 1 << 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ScalarType {
    F32,
    QuantizedI8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum WeightDType {
    F32,
    Int8,
}

impl WeightDType {
    fn to_scalar(self) -> ScalarType {
        match self {
            WeightDType::F32 => ScalarType::F32,
            WeightDType::Int8 => ScalarType::QuantizedI8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PipelineKey {
    dtype: ScalarType,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    use_subgroup: bool,
    use_f16: bool,
    use_bias: bool,
    fused_ops_mask: u32,
}

impl PipelineKey {
    fn new(
        dtype: ScalarType,
        tile: TileConfig,
        use_subgroup: bool,
        use_f16: bool,
        use_bias: bool,
        fused_ops_mask: u32,
    ) -> Self {
        Self {
            dtype,
            tile_m: tile.tile_m(),
            tile_n: tile.tile_n(),
            tile_k: tile.tile_k(),
            use_subgroup,
            use_f16,
            use_bias,
            fused_ops_mask,
        }
    }
}

struct PipelineEntry {
    shader: OnceLock<Arc<wgpu::ShaderModule>>,
    pipeline: OnceLock<Arc<ComputePipeline>>,
}

impl PipelineEntry {
    fn new() -> Self {
        Self {
            shader: OnceLock::new(),
            pipeline: OnceLock::new(),
        }
    }
}

struct PipelineCache {
    device: Arc<Device>,
    entries: Mutex<HashMap<PipelineKey, Arc<PipelineEntry>>>,
}

impl PipelineCache {
    fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            entries: Mutex::new(HashMap::new()),
        }
    }

    fn pipeline(&self, key: PipelineKey, layout: &PipelineLayout) -> Arc<ComputePipeline> {
        let mut guard = self.entries.lock().unwrap();
        let entry = guard
            .entry(key)
            .or_insert_with(|| Arc::new(PipelineEntry::new()))
            .clone();
        drop(guard);

        let shader = entry.shader.get_or_init(|| {
            let source = generate_matmul_shader(&key);
            let label = format!(
                "st.tensor.wgpu_dense.matmul_shader.{:?}.tile{}x{}x{}",
                key.dtype, key.tile_m, key.tile_n, key.tile_k
            );
            Arc::new(
                self.device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(&label),
                        source: wgpu::ShaderSource::Wgsl(source.into()),
                    }),
            )
        });

        entry
            .pipeline
            .get_or_init(|| {
                let label = format!(
                    "st.tensor.wgpu_dense.matmul_pipeline.{:?}.tile{}x{}x{}",
                    key.dtype, key.tile_m, key.tile_n, key.tile_k
                );
                Arc::new(
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(&label),
                            layout: Some(layout),
                            module: shader.as_ref(),
                            entry_point: "main",
                        }),
                )
            })
            .clone()
    }
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

fn div_ceil_usize(lhs: usize, rhs: usize) -> usize {
    if rhs == 0 {
        0
    } else {
        (lhs + rhs - 1) / rhs
    }
}

fn compute_rhs_tile_meta(
    inner: usize,
    cols: usize,
    tile: TileConfig,
) -> (usize, usize, usize, usize, usize) {
    let tile_k = tile.tile_k() as usize;
    let tile_n = tile.tile_n() as usize;
    let tiles_k = div_ceil_usize(inner, tile_k);
    let tiles_n = div_ceil_usize(cols, tile_n);
    let tile_elems = tile_k * tile_n;
    let total_tiles = tiles_k * tiles_n;
    let total_elems = total_tiles * tile_elems;
    (tile_k, tile_n, tiles_k, tiles_n, total_elems)
}

fn pack_rhs_f32(packed: &PackedB, tile: TileConfig) -> (Vec<f32>, usize) {
    let inner = packed.inner();
    let cols = packed.cols();
    let (tile_k, tile_n, tiles_k, tiles_n, total_elems) = compute_rhs_tile_meta(inner, cols, tile);
    let mut tiled = vec![0.0f32; total_elems];
    let slice = packed.as_slice();
    let layout = packed.layout();

    for tile_n_index in 0..tiles_n {
        for tile_k_index in 0..tiles_k {
            let tile_offset = (tile_n_index * tiles_k + tile_k_index) * tile_k * tile_n;
            for local_n in 0..tile_n {
                let col = tile_n_index * tile_n + local_n;
                for local_k in 0..tile_k {
                    let k = tile_k_index * tile_k + local_k;
                    let dst_index = tile_offset + local_n * tile_k + local_k;
                    if col < cols && k < inner {
                        let value = match layout {
                            PackedLayout::ColMajor => slice[col * inner + k],
                            PackedLayout::Tiled { .. } => slice[col * inner + k],
                        };
                        tiled[dst_index] = value;
                    }
                }
            }
        }
    }

    (tiled, total_elems)
}

fn pack_rhs_int8(packed: &PackedB, tile: TileConfig) -> (Vec<i32>, Vec<f32>, usize) {
    let inner = packed.inner();
    let cols = packed.cols();
    let (tile_k, tile_n, tiles_k, tiles_n, total_elems) = compute_rhs_tile_meta(inner, cols, tile);
    let mut quantized = Vec::with_capacity((total_elems + 3) / 4);
    let mut accumulator: u32 = 0;
    let mut lane: u32 = 0;
    let slice = packed.as_slice();
    let layout = packed.layout();

    let mut scales = vec![0.0f32; cols];
    for col in 0..cols {
        let mut max_abs = 0.0f32;
        for k in 0..inner {
            let value = match layout {
                PackedLayout::ColMajor => slice[col * inner + k],
                PackedLayout::Tiled { .. } => slice[col * inner + k],
            };
            let abs = value.abs();
            if abs > max_abs {
                max_abs = abs;
            }
        }
        scales[col] = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    }

    for tile_n_index in 0..tiles_n {
        for tile_k_index in 0..tiles_k {
            for local_n in 0..tile_n {
                let col = tile_n_index * tile_n + local_n;
                let scale = if col < cols { scales[col] } else { 1.0 };
                let inv_scale = 1.0 / scale;
                for local_k in 0..tile_k {
                    let k = tile_k_index * tile_k + local_k;
                    let mut quant = 0i32;
                    if col < cols && k < inner {
                        let raw = match layout {
                            PackedLayout::ColMajor => slice[col * inner + k],
                            PackedLayout::Tiled { .. } => slice[col * inner + k],
                        };
                        let scaled = (raw * inv_scale).round();
                        let clamped = scaled.clamp(-127.0, 127.0);
                        quant = clamped as i32;
                    }
                    let byte = (quant as i8) as u8 as u32;
                    accumulator |= byte << (lane * 8);
                    lane += 1;
                    if lane == 4 {
                        quantized.push(accumulator as i32);
                        accumulator = 0;
                        lane = 0;
                    }
                }
            }
        }
    }

    if lane != 0 {
        quantized.push(accumulator as i32);
    }

    (quantized, scales, total_elems)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum FusedActivation {
    Relu,
    Gelu,
}

struct GpuContext {
    context: WgpuContext,
    pipeline_cache: PipelineCache,
    weights_cache: Mutex<HashMap<RhsCacheKey, Weak<GpuPackedRhs>>>,
    bind_layout: Arc<BindGroupLayout>,
    pipeline_layout: Arc<PipelineLayout>,
    zero_storage: OnceLock<Arc<Buffer>>,
    zero_scales: OnceLock<Arc<Buffer>>,
    shader_f16: bool,
    supports_subgroup: bool,
    softmax_layout: BindGroupLayout,
    softmax_pipeline: Option<Arc<ComputePipeline>>,
    fused_attention: Option<FusedAttentionKernel>,
    fused_conv_layout: BindGroupLayout,
    fused_conv_pipeline_layout: PipelineLayout,
    fused_conv_pipelines: Mutex<HashMap<TileConfig, Arc<ComputePipeline>>>,
}

struct FusedAttentionKernel {
    layout: BindGroupLayout,
    pipeline: Arc<ComputePipeline>,
    max_head_dim: u32,
}

impl GpuContext {
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

        let adapter_features = adapter.features();
        let mut requested_features = wgpu::Features::empty();
        let shader_f16 = adapter_features.contains(wgpu::Features::SHADER_F16);
        if shader_f16 {
            requested_features |= wgpu::Features::SHADER_F16;
        }
        let supports_subgroup = false;

        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("st.tensor.wgpu_dense.device"),
                        required_features: requested_features,
                        required_limits: adapter.limits(),
                    },
                    None,
                )
                .await
        })
        .map_err(|err| err.to_string())?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let bind_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            },
        ));

        let pipeline_layout = Arc::new(device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.pipeline_layout"),
                bind_group_layouts: &[bind_layout.as_ref()],
                push_constant_ranges: &[],
            },
        ));

        let softmax_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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

        let softmax_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.softmax.pipeline_layout"),
                bind_group_layouts: &[&softmax_layout],
                push_constant_ranges: &[],
            });

        let softmax_pipeline = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("st.tensor.wgpu_dense.softmax"),
                source: wgpu::ShaderSource::Wgsl(ROW_SOFTMAX_WGSL.into()),
            });
            Arc::new(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("st.tensor.wgpu_dense.softmax"),
                    layout: Some(&softmax_pipeline_layout),
                    module: &shader,
                    entry_point: "main_cs",
                }),
            )
        }))
        .ok();

        let fused_attention = std::panic::catch_unwind(AssertUnwindSafe(|| {
            let shader_source = FUSED_ATTENTION_WGSL_TEMPLATE
                .replace("{WORKGROUP_SIZE}", &FUSED_ATTENTION_WORKGROUP.to_string())
                .replace("{MAX_HEAD_DIM}", &FUSED_ATTENTION_MAX_HEAD_DIM.to_string());
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_attention.shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_attention.layout"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
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
                label: Some("st.tensor.wgpu_dense.fused_attention.pipeline_layout"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });
            let pipeline = Arc::new(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("st.tensor.wgpu_dense.fused_attention"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: "main",
                },
            ));
            FusedAttentionKernel {
                layout,
                pipeline,
                max_head_dim: FUSED_ATTENTION_MAX_HEAD_DIM,
            }
        }))
        .ok();

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
                label: Some("st.tensor.wgpu_dense.fused_conv.pipeline_layout"),
                bind_group_layouts: &[&fused_conv_layout],
                push_constant_ranges: &[],
            });

        Ok(Self {
            context: WgpuContext::new(device.clone(), queue.clone()),
            pipeline_cache: PipelineCache::new(device.clone()),
            weights_cache: Mutex::new(HashMap::new()),
            bind_layout,
            pipeline_layout,
            zero_storage: OnceLock::new(),
            zero_scales: OnceLock::new(),
            shader_f16,
            supports_subgroup,
            softmax_layout,
            softmax_pipeline,
            fused_attention,
            fused_conv_layout,
            fused_conv_pipeline_layout,
            fused_conv_pipelines: Mutex::new(HashMap::new()),
        })
    }

    fn device(&self) -> &Device {
        self.context.device()
    }

    fn queue(&self) -> &Queue {
        self.context.queue()
    }

    fn bind_layout(&self) -> &BindGroupLayout {
        self.bind_layout.as_ref()
    }

    fn pipeline_layout(&self) -> &PipelineLayout {
        self.pipeline_layout.as_ref()
    }

    fn zero_storage_buffer(&self) -> Arc<Buffer> {
        self.zero_storage
            .get_or_init(|| {
                Arc::new(
                    self.device()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("st.tensor.wgpu_dense.zeros"),
                            contents: bytemuck::cast_slice(&[0.0f32]),
                            usage: wgpu::BufferUsages::STORAGE,
                        }),
                )
            })
            .clone()
    }

    fn zero_scales_buffer(&self) -> Arc<Buffer> {
        self.zero_scales
            .get_or_init(|| {
                Arc::new(
                    self.device()
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("st.tensor.wgpu_dense.zero_scales"),
                            contents: bytemuck::cast_slice(&[1.0f32]),
                            usage: wgpu::BufferUsages::STORAGE,
                        }),
                )
            })
            .clone()
    }

    fn supports_softmax(&self) -> bool {
        self.softmax_pipeline.is_some()
    }

    fn softmax_pipeline(&self) -> Option<Arc<ComputePipeline>> {
        self.softmax_pipeline.as_ref().map(Arc::clone)
    }

    fn rhs_from_packed(
        &self,
        packed: &PackedB,
        tile: TileConfig,
    ) -> Result<Arc<GpuPackedRhs>, String> {
        if packed.inner() == 0 || packed.cols() == 0 {
            return Err("packed matrix dimensions must be positive".into());
        }

        let dtype = if should_quantize(packed.inner(), packed.cols()) {
            ScalarType::QuantizedI8
        } else {
            ScalarType::F32
        };
        let key = RhsCacheKey::new(packed, tile, dtype);
        if let Some(existing) = self
            .weights_cache
            .lock()
            .unwrap()
            .get(&key)
            .and_then(|weak| weak.upgrade())
        {
            return Ok(existing);
        }

        let (buffer, scales) = match dtype {
            ScalarType::QuantizedI8 => {
                let (quantized, scales_vec, _total) = pack_rhs_int8(packed, tile);
                let buffer = Arc::new(self.device().create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("st.tensor.wgpu_dense.packed_rhs.int8"),
                        contents: bytemuck::cast_slice(&quantized),
                        usage: wgpu::BufferUsages::STORAGE,
                    },
                ));
                let scales = Arc::new(self.device().create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("st.tensor.wgpu_dense.packed_rhs.scales"),
                        contents: bytemuck::cast_slice(&scales_vec),
                        usage: wgpu::BufferUsages::STORAGE,
                    },
                ));
                (buffer, Some(scales))
            }
            ScalarType::F32 => {
                let (values, _total) = pack_rhs_f32(packed, tile);
                let buffer = Arc::new(self.device().create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("st.tensor.wgpu_dense.packed_rhs.f32"),
                        contents: bytemuck::cast_slice(&values),
                        usage: wgpu::BufferUsages::STORAGE,
                    },
                ));
                (buffer, None)
            }
        };

        let prepared = Arc::new(GpuPackedRhs {
            tile,
            dtype,
            buffer,
            scales,
            _cols: packed.cols(),
            _inner: packed.inner(),
        });

        self.weights_cache
            .lock()
            .unwrap()
            .insert(key, Arc::downgrade(&prepared));
        Ok(prepared)
    }

    fn fused_attention_kernel(&self) -> Option<&FusedAttentionKernel> {
        self.fused_attention.as_ref()
    }

    fn fused_attention_bind_group(
        &self,
        kernel: &FusedAttentionKernel,
        queries: &Buffer,
        keys: &Buffer,
        values: &Buffer,
        z_bias: &Buffer,
        attn_bias: &Buffer,
        output: &Buffer,
        params: &Buffer,
    ) -> BindGroup {
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_attention.bind_group"),
            layout: &kernel.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: queries.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: values.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: z_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: attn_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    fn softmax_bind_group(&self, input: &Buffer, output: &Buffer, params: &Buffer) -> BindGroup {
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax.bind_group"),
            layout: &self.softmax_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    fn fused_conv_pipeline_for(&self, config: TileConfig) -> Arc<ComputePipeline> {
        let mut pipelines = self.fused_conv_pipelines.lock().unwrap();
        if let Some(pipeline) = pipelines.get(&config) {
            return pipeline.clone();
        }

        let shader_source = instantiate_tile_template(FUSED_CONV_WGSL_TEMPLATE, config);
        let shader_label = format!(
            "st.tensor.wgpu_dense.fused_conv_shader.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let shader = self
            .device()
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
        let pipeline = Arc::new(self.device().create_compute_pipeline(
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

    fn fused_conv_bind_group(
        &self,
        input: &Buffer,
        weights: &Buffer,
        output: &Buffer,
        params: &Buffer,
    ) -> BindGroup {
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_conv.bind_group"),
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

fn instantiate_tile_template(template: &str, config: TileConfig) -> String {
    let tile_m = config.tile_m();
    let tile_n = config.tile_n();
    let tile_k = config.tile_k();
    let tile_mk = tile_m * tile_k;
    let tile_nk = tile_n * tile_k;

    template
        .replace("{tile_m}", &tile_m.to_string())
        .replace("{tile_n}", &tile_n.to_string())
        .replace("{tile_k}", &tile_k.to_string())
        .replace("{tile_mk}", &(tile_mk.to_string() + "u"))
        .replace("{tile_nk}", &(tile_nk.to_string() + "u"))
}

static CONTEXT: OnceLock<Arc<GpuContext>> = OnceLock::new();

fn dense_context() -> Result<Arc<GpuContext>, String> {
    if let Some(ctx) = CONTEXT.get() {
        return Ok(ctx.clone());
    }
    let ctx = Arc::new(GpuContext::new()?);
    let _ = CONTEXT.set(ctx.clone());
    Ok(ctx)
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulUniforms {
    rows: u32,
    cols: u32,
    inner: u32,
    flags: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RowSoftmaxParams {
    rows: u32,
    cols: u32,
    in_stride: u32,
    out_stride: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedAttentionParams {
    contexts: u32,
    sequence: u32,
    head_dim: u32,
    flags: u32,
    scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
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
    _pad2: u32,
    _pad3: u32,
}

struct WeightBuffers {
    buffer: Arc<Buffer>,
    dtype: ScalarType,
    scales: Option<Arc<Buffer>>,
}

impl WeightBuffers {
    fn as_binding(&self) -> (&Buffer, ScalarType, Option<&Buffer>) {
        (
            self.buffer.as_ref(),
            self.dtype,
            self.scales.as_ref().map(Arc::as_ref),
        )
    }
}

fn generate_matmul_shader(key: &PipelineKey) -> String {
    let enable_f16 = if key.use_f16 {
        "enable f16;
"
    } else {
        ""
    };
    let rhs_storage_type = match key.dtype {
        ScalarType::F32 => "array<f32>".to_string(),
        ScalarType::QuantizedI8 => "array<u32>".to_string(),
    };

    let rhs_load_body = match key.dtype {
        ScalarType::F32 => "return rhs_packed[k * params.cols + col];".to_string(),
        ScalarType::QuantizedI8 => "let stride = (params.inner + 3u) / 4u;
        let base = col * stride + (k >> 2u);
        let word = rhs_packed[base];
        let lane = (k & 3u) * 8u;
        let byte_val = (word >> lane) & 0xFFu;
        let signed = bitcast<i32>(byte_val << 24u) >> 24;
        let scale = scales[col];
        return f32(signed) * scale;"
            .to_string(),
    };

    let fma_line = if key.use_f16 {
        "let lhs16 = f16(lhs_val);
            let rhs16 = f16(rhs_val);
            acc = acc + f32(lhs16 * rhs16);"
    } else {
        "acc = acc + lhs_val * rhs_val;"
    };

    MATMUL_WGSL_TEMPLATE
        .replace("{f16_enable}", enable_f16)
        .replace("{rhs_storage_type}", &rhs_storage_type)
        .replace("{rhs_load_body}", &rhs_load_body)
        .replace("{tile_m}", &key.tile_m.to_string())
        .replace("{tile_n}", &key.tile_n.to_string())
        .replace("{tile_k}", &key.tile_k.to_string())
        .replace("{workgroup_size_x}", &key.tile_n.to_string())
        .replace("{workgroup_size_y}", &key.tile_m.to_string())
        .replace("{fma_line}", fma_line)
}

fn should_quantize(inner: usize, cols: usize) -> bool {
    inner.saturating_mul(cols) >= QUANTIZATION_MIN_VOLUME
}

struct QuantizedWeights {
    packed: Vec<u32>,
    scales: Vec<f32>,
}

impl QuantizedWeights {
    fn from_f32(weights: &[f32], inner: usize, cols: usize) -> Self {
        let stride = (inner + 3) / 4;
        let mut packed = vec![0u32; cols * stride];
        let mut scales = vec![0.0f32; cols];
        for col in 0..cols {
            let mut max_abs = 0.0f32;
            for k in 0..inner {
                let value = weights[k * cols + col];
                max_abs = max_abs.max(value.abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            scales[col] = scale;
            for pack_idx in 0..stride {
                let mut word = 0u32;
                for lane in 0..4 {
                    let k = pack_idx * 4 + lane;
                    let quant = if scale == 0.0 || k >= inner {
                        0i32
                    } else {
                        let value = weights[k * cols + col] / scale;
                        value.round().clamp(-128.0, 127.0) as i32
                    };
                    let byte = (quant as u32) & 0xFF;
                    word |= byte << (lane * 8);
                }
                packed[col * stride + pack_idx] = word;
            }
        }
        Self { packed, scales }
    }
}

#[derive(Clone)]
struct GpuPackedRhs {
    tile: TileConfig,
    dtype: ScalarType,
    buffer: Arc<Buffer>,
    scales: Option<Arc<Buffer>>,
    _cols: usize,
    _inner: usize,
}

impl GpuPackedRhs {
    fn buffer(&self) -> Arc<Buffer> {
        Arc::clone(&self.buffer)
    }

    fn scales(&self) -> Option<Arc<Buffer>> {
        self.scales.as_ref().map(Arc::clone)
    }

    fn dtype(&self) -> ScalarType {
        self.dtype
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RhsCacheKey {
    data_ptr: usize,
    cols: usize,
    inner: usize,
    tile: TileConfig,
    dtype: ScalarType,
}

impl RhsCacheKey {
    fn new(packed: &PackedB, tile: TileConfig, dtype: ScalarType) -> Self {
        Self {
            data_ptr: packed.as_slice().as_ptr() as usize,
            cols: packed.cols(),
            inner: packed.inner(),
            tile,
            dtype,
        }
    }
}

fn upload_weights(device: &Device, rhs: &[f32], inner: usize, cols: usize) -> WeightBuffers {
    if should_quantize(inner, cols) {
        let quantized = QuantizedWeights::from_f32(rhs, inner, cols);
        let buffer = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("st.tensor.wgpu_dense.rhs.quantized"),
                contents: bytemuck::cast_slice(quantized.packed.as_slice()),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        let scales = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("st.tensor.wgpu_dense.rhs.scales"),
                contents: bytemuck::cast_slice(quantized.scales.as_slice()),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        WeightBuffers {
            buffer,
            dtype: ScalarType::QuantizedI8,
            scales: Some(scales),
        }
    } else {
        let buffer = Arc::new(
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("st.tensor.wgpu_dense.rhs"),
                contents: bytemuck::cast_slice(rhs),
                usage: wgpu::BufferUsages::STORAGE,
            }),
        );
        WeightBuffers {
            buffer,
            dtype: ScalarType::F32,
            scales: None,
        }
    }
}

fn create_bind_group(
    ctx: &GpuContext,
    key: &PipelineKey,
    lhs: &Buffer,
    rhs_buffer: &Buffer,
    out: &Buffer,
    bias: Option<&Buffer>,
    residual: Option<&Buffer>,
    scales: Option<&Buffer>,
    params: &Buffer,
) -> BindGroup {
    let zero_storage_arc = ctx.zero_storage_buffer();
    let zero_scales_arc = ctx.zero_scales_buffer();
    let bias_binding = bias.unwrap_or(zero_storage_arc.as_ref());
    let residual_binding = residual.unwrap_or(zero_storage_arc.as_ref());
    let scales_binding = if matches!(key.dtype, ScalarType::QuantizedI8) {
        scales.unwrap_or(zero_scales_arc.as_ref())
    } else {
        zero_scales_arc.as_ref()
    };
    ctx.device().create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.tensor.wgpu_dense.bind_group"),
        layout: ctx.bind_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: lhs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: rhs_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: bias_binding.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: residual_binding.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: scales_binding.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: params.as_entire_binding(),
            },
        ],
    })
}

fn dispatch_matmul(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    lhs: &Buffer,
    rhs: &WeightBuffers,
    out: &Buffer,
    rows: usize,
    inner: usize,
    cols: usize,
    tile: TileConfig,
    use_bias: bool,
    fused_ops_mask: u32,
    bias: Option<&Buffer>,
    residual: Option<&Buffer>,
) {
    let (rhs_buffer, dtype, scales) = rhs.as_binding();
    let use_f16 = ctx.shader_f16;
    let key = PipelineKey::new(
        dtype,
        tile,
        ctx.supports_subgroup,
        use_f16,
        use_bias,
        fused_ops_mask,
    );
    let mut flags = fused_ops_mask;
    if use_bias {
        flags |= FLAG_USE_BIAS;
    }
    if matches!(dtype, ScalarType::QuantizedI8) {
        flags |= FLAG_USE_INT8;
    }
    if use_f16 {
        flags |= FLAG_USE_F16;
    }
    let params = MatmulUniforms {
        rows: rows as u32,
        cols: cols as u32,
        inner: inner as u32,
        flags,
    };
    let params_buf = ctx
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.tensor.wgpu_dense.matmul.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bind_group = create_bind_group(
        ctx,
        &key,
        lhs,
        rhs_buffer,
        out,
        bias,
        residual,
        scales,
        &params_buf,
    );
    let pipeline = ctx.pipeline_cache.pipeline(key, ctx.pipeline_layout());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.matmul.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        let groups_x = ((cols as u32) + tile.tile_n() - 1) / tile.tile_n();
        let groups_y = ((rows as u32) + tile.tile_m() - 1) / tile.tile_m();
        pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), 1);
    }
}

#[allow(dead_code)]
fn dispatch_matmul_with_options(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    lhs: &Buffer,
    rhs: &Buffer,
    bias: Option<&Buffer>,
    residual: Option<&Buffer>,
    scales: Option<&Buffer>,
    out: &Buffer,
    rows: usize,
    inner: usize,
    cols: usize,
    tile: TileConfig,
    fused_ops_mask: u32,
    dtype: WeightDType,
) {
    let rhs_buffers = WeightBuffers {
        buffer: Arc::new(rhs.clone()),
        dtype: dtype.to_scalar(),
        scales: scales.map(|buf| Arc::new(buf.clone())),
    };
    let bias_arc = bias.map(|buf| Arc::new(buf.clone()));
    let residual_arc = residual.map(|buf| Arc::new(buf.clone()));
    dispatch_matmul(
        ctx,
        encoder,
        lhs,
        &rhs_buffers,
        out,
        rows,
        inner,
        cols,
        tile,
        bias.is_some(),
        fused_ops_mask,
        bias_arc.as_ref().map(Arc::as_ref),
        residual_arc.as_ref().map(Arc::as_ref),
    );
}

#[allow(dead_code)]
fn dispatch_fused_linear(
    ctx: &GpuContext,
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
    let mask = match activation {
        FusedActivation::Relu => FUSED_OP_RELU,
        FusedActivation::Gelu => FUSED_OP_GELU,
    };
    dispatch_matmul_with_options(
        ctx,
        encoder,
        lhs,
        rhs,
        Some(bias),
        None,
        None,
        out,
        rows,
        inner,
        cols,
        tile,
        mask,
        WeightDType::F32,
    );
}

#[allow(dead_code)]
fn dispatch_fused_linear_with_residual(
    ctx: &GpuContext,
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
    let mut mask = match activation {
        FusedActivation::Relu => FUSED_OP_RELU,
        FusedActivation::Gelu => FUSED_OP_GELU,
    };
    mask |= FUSED_OP_RESIDUAL;
    dispatch_matmul_with_options(
        ctx,
        encoder,
        lhs,
        rhs,
        Some(bias),
        Some(residual),
        None,
        out,
        rows,
        inner,
        cols,
        tile,
        mask,
        WeightDType::F32,
    );
}

fn upload_lhs(device: &Device, label: &str, data: &[f32]) -> Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE,
    })
}

fn allocate_output(device: &Device, label: &str, elements: usize) -> Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (elements * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
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

    let lhs_buf = upload_lhs(device, "st.tensor.wgpu_dense.matmul.lhs", a);
    let rhs_buf = upload_weights(device, b, inner, cols);
    let out_buf = allocate_output(device, "st.tensor.wgpu_dense.matmul.out", rows * cols);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.matmul.encoder"),
    });
    let tile = select_tile_config(rows, inner, cols);
    dispatch_matmul(
        &ctx,
        &mut encoder,
        &lhs_buf,
        &rhs_buf,
        &out_buf,
        rows,
        inner,
        cols,
        tile,
        false,
        0,
        None,
        None,
    );
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn matmul_prepacked(
    lhs: &[f32],
    packed_rhs: &PackedB,
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
    if packed_rhs.inner() != inner {
        return Err("packed rhs inner dimension mismatch".into());
    }
    if packed_rhs.cols() != cols {
        return Err("packed rhs column dimension mismatch".into());
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let lhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.prepacked.lhs"),
        contents: bytemuck::cast_slice(lhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = allocate_output(device, "st.tensor.wgpu_dense.prepacked.out", rows * cols);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.prepacked.encoder"),
    });
    let tile = select_tile_config(rows, inner, cols);
    let weights = ctx.rhs_from_packed(packed_rhs, tile)?;
    let rhs_buffers = WeightBuffers {
        buffer: weights.buffer(),
        dtype: weights.dtype(),
        scales: weights.scales(),
    };
    dispatch_matmul(
        &ctx,
        &mut encoder,
        &lhs_buf,
        &rhs_buffers,
        &out_buf,
        rows,
        inner,
        cols,
        tile,
        false,
        0,
        None,
        None,
    );
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn matmul_bias_relu(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    matmul_with_bias_activation(
        lhs,
        rhs,
        bias,
        rows,
        inner,
        cols,
        Some(FusedActivation::Relu),
        None,
    )
}

pub fn matmul_bias_gelu(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    matmul_with_bias_activation(
        lhs,
        rhs,
        bias,
        rows,
        inner,
        cols,
        Some(FusedActivation::Gelu),
        None,
    )
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
    matmul_with_bias_activation(
        lhs,
        rhs,
        bias,
        rows,
        inner,
        cols,
        Some(FusedActivation::Relu),
        Some(residual),
    )
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
    matmul_with_bias_activation(
        lhs,
        rhs,
        bias,
        rows,
        inner,
        cols,
        Some(FusedActivation::Gelu),
        Some(residual),
    )
}

fn matmul_with_bias_activation(
    lhs: &[f32],
    rhs: &[f32],
    bias: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
    activation: Option<FusedActivation>,
    residual: Option<&[f32]>,
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
    if let Some(residual) = residual {
        if residual.len() != rows * cols {
            return Err(format!(
                "residual length mismatch: expected {} elements, got {}",
                rows * cols,
                residual.len()
            ));
        }
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let lhs_buf = upload_lhs(device, "st.tensor.wgpu_dense.fused.lhs", lhs);
    let rhs_buf = upload_weights(device, rhs, inner, cols);
    let bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.fused.bias"),
        contents: bytemuck::cast_slice(bias),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let residual_buf = residual.map(|data| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.tensor.wgpu_dense.fused.residual"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    });
    let out_buf = allocate_output(device, "st.tensor.wgpu_dense.fused.out", rows * cols);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.fused.encoder"),
    });
    let tile = select_tile_config(rows, inner, cols);
    let mut fused_mask = 0u32;
    if let Some(act) = activation {
        match act {
            FusedActivation::Relu => fused_mask |= FLAG_FUSED_RELU,
            FusedActivation::Gelu => fused_mask |= FLAG_FUSED_GELU,
        }
    }
    if residual_buf.is_some() {
        fused_mask |= FLAG_FUSED_RESIDUAL;
    }
    dispatch_matmul(
        &ctx,
        &mut encoder,
        &lhs_buf,
        &rhs_buf,
        &out_buf,
        rows,
        inner,
        cols,
        tile,
        true,
        fused_mask,
        Some(&bias_buf),
        residual_buf.as_ref().map(|buf| buf),
    );
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn row_softmax(input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    if rows == 0 || cols == 0 {
        return Err("matrix dimensions must be positive".into());
    }
    if input.len() != rows * cols {
        return Err(format!(
            "input length mismatch: expected {} elements, got {}",
            rows * cols,
            input.len()
        ));
    }

    let rows_u32 = u32::try_from(rows).map_err(|_| "rows exceed u32::MAX".to_string())?;
    let cols_u32 = u32::try_from(cols).map_err(|_| "cols exceed u32::MAX".to_string())?;

    let ctx = dense_context()?;
    if !ctx.supports_softmax() {
        return Err("wgpu device lacks subgroup row softmax support".into());
    }

    let device = ctx.device();
    let queue = ctx.queue();

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.softmax.input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buf = allocate_output(device, "st.tensor.wgpu_dense.softmax.output", rows * cols);
    let params = RowSoftmaxParams {
        rows: rows_u32,
        cols: cols_u32,
        in_stride: cols_u32,
        out_stride: cols_u32,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.softmax.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.softmax_bind_group(&input_buf, &output_buf, &params_buf);
    let pipeline = ctx
        .softmax_pipeline()
        .ok_or_else(|| "row softmax pipeline unavailable".to_string())?;

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.softmax.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows_u32, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &output_buf, rows * cols)
}

pub fn is_available() -> bool {
    dense_context().is_ok()
}

pub fn supports_row_softmax(rows: usize, cols: usize) -> bool {
    if rows == 0 || cols == 0 {
        return false;
    }
    if rows > u32::MAX as usize || cols > u32::MAX as usize {
        return false;
    }
    if let Ok(ctx) = dense_context() {
        ctx.supports_softmax()
    } else {
        false
    }
}

pub fn fused_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    contexts: usize,
    sequence: usize,
    head_dim: usize,
    scale: f32,
    z_bias: Option<&[f32]>,
    attn_bias: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    if contexts == 0 || sequence == 0 || head_dim == 0 {
        return Err("attention dimensions must be positive".into());
    }

    let volume = contexts
        .checked_mul(sequence)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| "attention volume exceeds usize range".to_string())?;

    if queries.len() != volume {
        return Err(format!(
            "query buffer length mismatch: expected {} elements, got {}",
            volume,
            queries.len()
        ));
    }
    if keys.len() != volume {
        return Err(format!(
            "key buffer length mismatch: expected {} elements, got {}",
            volume,
            keys.len()
        ));
    }
    if values.len() != volume {
        return Err(format!(
            "value buffer length mismatch: expected {} elements, got {}",
            volume,
            values.len()
        ));
    }

    if let Some(bias) = z_bias {
        let expected = contexts * sequence;
        if bias.len() != expected {
            return Err(format!(
                "z-bias length mismatch: expected {} elements, got {}",
                expected,
                bias.len()
            ));
        }
    }

    if let Some(bias) = attn_bias {
        let expected = contexts
            .checked_mul(sequence)
            .and_then(|v| v.checked_mul(sequence))
            .ok_or_else(|| "attention bias volume exceeds usize range".to_string())?;
        if bias.len() != expected {
            return Err(format!(
                "attention bias length mismatch: expected {} elements, got {}",
                expected,
                bias.len()
            ));
        }
    }

    if contexts > u32::MAX as usize || sequence > u32::MAX as usize || head_dim > u32::MAX as usize
    {
        return Err("attention dimensions exceed u32 dispatch range".into());
    }

    let ctx = dense_context()?;
    let kernel = ctx
        .fused_attention_kernel()
        .ok_or_else(|| "fused attention kernel unavailable on this device".to_string())?;

    if (head_dim as u32) > kernel.max_head_dim {
        return Err(format!(
            "head dimension {} exceeds templated maximum {}",
            head_dim, kernel.max_head_dim
        ));
    }

    let device = ctx.device();
    let queue = ctx.queue();

    let query_buf = upload_lhs(device, "st.tensor.wgpu_dense.attn.queries", queries);
    let key_buf = upload_lhs(device, "st.tensor.wgpu_dense.attn.keys", keys);
    let value_buf = upload_lhs(device, "st.tensor.wgpu_dense.attn.values", values);
    let output_buf = allocate_output(device, "st.tensor.wgpu_dense.attn.output", volume);

    let z_bias_buf =
        z_bias.map(|data| upload_lhs(device, "st.tensor.wgpu_dense.attn.z_bias", data));
    let attn_bias_buf =
        attn_bias.map(|data| upload_lhs(device, "st.tensor.wgpu_dense.attn.attn_bias", data));
    let zero_storage = ctx.zero_storage_buffer();
    let z_binding = z_bias_buf
        .as_ref()
        .map(|buf| buf)
        .unwrap_or_else(|| zero_storage.as_ref());
    let attn_binding = attn_bias_buf
        .as_ref()
        .map(|buf| buf)
        .unwrap_or_else(|| zero_storage.as_ref());

    let flags = {
        let mut mask = 0u32;
        if z_bias.is_some() {
            mask |= FUSED_ATTENTION_FLAG_USE_Z_BIAS;
        }
        if attn_bias.is_some() {
            mask |= FUSED_ATTENTION_FLAG_USE_ATTN_BIAS;
        }
        mask
    };

    let params = FusedAttentionParams {
        contexts: contexts as u32,
        sequence: sequence as u32,
        head_dim: head_dim as u32,
        flags,
        scale,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.attn.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = ctx.fused_attention_bind_group(
        kernel,
        &query_buf,
        &key_buf,
        &value_buf,
        z_binding,
        attn_binding,
        &output_buf,
        &params_buf,
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.attn.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.attn.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(kernel.pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(sequence as u32, contexts as u32, 1);
    }
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &output_buf, volume)
}

pub fn supports_fused_attention(contexts: usize, sequence: usize, head_dim: usize) -> bool {
    if contexts == 0 || sequence == 0 || head_dim == 0 {
        return false;
    }
    if contexts > u32::MAX as usize || sequence > u32::MAX as usize || head_dim > u32::MAX as usize
    {
        return false;
    }
    if let Ok(ctx) = dense_context() {
        ctx.fused_attention_kernel()
            .filter(|kernel| (head_dim as u32) <= kernel.max_head_dim)
            .is_some()
    } else {
        false
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
    bias: Option<&[f32]>,
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
    if let Some(bias) = bias {
        if bias.len() != out_channels {
            return Err("bias length mismatch".into());
        }
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
        _pad2: 0,
        _pad3: 0,
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
    let output_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.conv.output",
        rows * out_channels,
    );

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
