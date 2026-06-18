// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use crate::backend::wgpu_util::WgpuContext;
use crate::pure::{
    spiral_softmax_hardmax_consensus, Layout, PackedB, PackedLayout, SpiralConsensusStats,
};
use crate::util::readback_f32;
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use st_kdsl::autotune_store::{load_best_typed, lookup_similar, record_best};
use st_kdsl::{
    AutotuneKey, AutotuneRegistry, DeviceProfile, KernelProfile, TelemetrySample, TelemetrySummary,
};
use std::any::Any;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::convert::TryFrom;
use std::env;
use std::f32::consts::PI;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;
use wgpu::{
    AdapterInfo, BindGroup, BindGroupLayout, Buffer, ComputePipeline, Device, PipelineLayout, Queue,
};

const MATMUL_WGSL_TEMPLATE: &str = include_str!("../wgpu_shaders/dense_matmul.wgsl");
const FUSED_CONV_WGSL_TEMPLATE: &str = include_str!("../wgpu_shaders/fused_im2col_matmul.wgsl");
const FUSED_GRAD_INPUT_WGSL_TEMPLATE: &str =
    include_str!("../wgpu_shaders/fused_grad_input_col2im.wgsl");
const FUSED_GELU_BACK_WGSL_TEMPLATE: &str = include_str!("../wgpu_shaders/fused_gelu_back.wgsl");
const REDUCE_DB_WGSL_TEMPLATE: &str = include_str!("../wgpu_shaders/reduce_db.wgsl");
const LAYER_NORM_WGSL: &str = include_str!("../wgpu_shaders/layer_norm.wgsl");
const TENSOR_UTILS_WGSL: &str = include_str!("../wgpu_shaders/tensor_utils.wgsl");
const RAMANUJAN_PI_WGSL: &str = include_str!("../wgpu_shaders/ramanujan_pi.wgsl");
const LSTM_BACKWARD_SCAN_WGSL: &str = include_str!("../wgpu_shaders/lstm_backward_scan.wgsl");
const ROW_SOFTMAX_WGSL: &str =
    include_str!("../../../st-backend-wgpu/src/shaders/softmax_workgroup.wgsl");
const ROW_SOFTMAX_SUBGROUP_WGSL: &str =
    include_str!("../../../st-backend-wgpu/src/shaders/softmax_subgroup.wgsl");
const SOFTMAX_ZSPACE_WGSL: &str =
    include_str!("../../../st-backend-wgpu/src/shaders/softmax_zspace_projection.wgsl");
const SOFTMAX_SPIRAL_WGSL: &str =
    include_str!("../../../st-backend-wgpu/src/shaders/softmax_spiral_consensus.wgsl");
const FUSED_ATTENTION_WGSL_TEMPLATE: &str =
    include_str!("../../../st-backend-wgpu/src/shaders/fused_attention_online.wgsl");

const FUSED_ATTENTION_WORKGROUP: u32 = 128;
const FUSED_ATTENTION_MAX_HEAD_DIM: u32 = 256;
const FUSED_GELU_BACK_WG_ROWS: u32 = 16;
const FUSED_GELU_BACK_WG_COLS: u32 = 16;
const REDUCE_DB_WORKGROUP: u32 = 256;
const LAYER_NORM_WORKGROUP: u32 = 256;
const TENSOR_UTIL_WORKGROUP: u32 = 256;
const LSTM_BACKWARD_SCAN_WORKGROUP: u32 = 64;

const FLAG_USE_BIAS: u32 = 1 << 0;
const FLAG_FUSED_RELU: u32 = 1 << 1;
const FLAG_FUSED_GELU: u32 = 1 << 2;
const FLAG_FUSED_RESIDUAL: u32 = 1 << 3;
const FLAG_USE_INT8: u32 = 1 << 4;
const FLAG_USE_F16: u32 = 1 << 5;
const FLAG_LHS_TRANSPOSE: u32 = 1 << 6;

const FUSED_OP_RELU: u32 = FLAG_FUSED_RELU;
const FUSED_OP_GELU: u32 = FLAG_FUSED_GELU;
const FUSED_OP_RESIDUAL: u32 = FLAG_FUSED_RESIDUAL;

const QUANTIZATION_MIN_VOLUME: usize = 64 * 64;

const FUSED_ATTENTION_FLAG_USE_Z_BIAS: u32 = 1 << 0;
const FUSED_ATTENTION_FLAG_USE_ATTN_BIAS: u32 = 1 << 1;
const LAYER_NORM_FLAG_USE_RESIDUAL: u32 = 1 << 0;

const GRAD_INPUT_TILE_X: u32 = 4;
const GRAD_INPUT_TILE_Y: u32 = 4;
const GRAD_INPUT_TILE_Z: u32 = 4;
const RAMANUJAN_PI_ITERATIONS: usize = 6;

const SOFTMAX_WORKGROUP_SIZE: f32 = 256.0;
const SOFTMAX_FLOPS_PER_ELEMENT: f64 = 5.0;
const SOFTMAX_BYTES_PER_ELEMENT: f64 = 12.0;
const GOLDEN_RATIO: f32 = 1.618_034;
const GOLDEN_RATIO_CONJUGATE: f32 = 0.618_034;
const GOLDEN_RATIO_BIAS: f32 = 0.381_966;
const GOLDEN_ANGLE_DEG: f32 = 137.507_77;
const GOLDEN_ANGLE_RAD: f32 = GOLDEN_ANGLE_DEG * (PI / 180.0);
const ZSPACE_MIN_ENERGY: f32 = 1e-6;
const LEECH_PACKING_DENSITY: f64 = 0.001_929_574_309_403_922_5;
const SOFTMAX_ZSPACE_LEECH_RANK: usize = 24;
const SOFTMAX_ZSPACE_LEECH_WEIGHT: f64 = 0.75;
const SOFTMAX_ZSPACE_RAMANUJAN_ITERS: usize = 6;
const SOFTMAX_ZSPACE_LEECH_SCALE: f64 = 0.05;
const SPIRAL_PROJECTOR_RANK: usize = 24;
const SPIRAL_PROJECTOR_WEIGHT: f64 = 0.75;
const SPIRAL_PROJECTOR_RAMANUJAN_ITERS: usize = 6;
const SPIRAL_LEECH_PACKING_DENSITY: f64 = 0.001_929_574_309_403_922_5;
const SPIRAL_ENTROPY_EPSILON: f32 = 1e-7;

pub type FusedGeluBackwardOutput = (Vec<f32>, Vec<f32>, Vec<f32>);
pub type RowSoftmaxHardmaxSpiralOutput = (Vec<f32>, Vec<f32>, Vec<f32>, SpiralConsensusStats);

fn global_autotune_registry() -> &'static Mutex<AutotuneRegistry> {
    static REGISTRY: OnceLock<Mutex<AutotuneRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(AutotuneRegistry::new()))
}

fn ramanujan_pi(iterations: usize) -> f64 {
    let iterations = iterations.max(1);
    let mut sum = 0.0f64;
    let mut factor = 1.0f64;
    let base = 396f64.powi(4);
    let prefactor = (2.0 * 2.0f64.sqrt()) / 9801.0;
    for k in 0..iterations {
        let kf = k as f64;
        sum += factor * (1103.0 + 26390.0 * kf);
        if k + 1 < iterations {
            let next = kf + 1.0;
            let numerator =
                (4.0 * next - 3.0) * (4.0 * next - 2.0) * (4.0 * next - 1.0) * (4.0 * next);
            let denominator = next.powi(4) * base;
            factor *= numerator / denominator;
        }
    }
    (prefactor * sum).recip()
}

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

type ShaderCacheEntry = Result<Arc<wgpu::ShaderModule>, String>;
type PipelineCacheEntry = Result<Arc<ComputePipeline>, String>;

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
    shader: OnceLock<ShaderCacheEntry>,
    pipeline: OnceLock<PipelineCacheEntry>,
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

    fn pipeline(
        &self,
        key: PipelineKey,
        layout: &PipelineLayout,
    ) -> Result<Arc<ComputePipeline>, anyhow::Error> {
        let mut guard = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let entry = guard
            .entry(key)
            .or_insert_with(|| Arc::new(PipelineEntry::new()))
            .clone();
        drop(guard);

        let device = self.device.clone();
        let shader_key = key;
        let shader_result = entry.shader.get_or_init(|| {
            let source = generate_matmul_shader(&shader_key);
            let label = format!(
                "st.tensor.wgpu_dense.matmul_shader.{:?}.tile{}x{}x{}",
                shader_key.dtype, shader_key.tile_m, shader_key.tile_n, shader_key.tile_k
            );
            create_wgsl_module(device.as_ref(), &label, source.as_str())
                .map(Arc::new)
                .map_err(|err| err.to_string())
        });
        let shader = match shader_result {
            Ok(shader) => Arc::clone(shader),
            Err(err) => return Err(anyhow!(err.clone())),
        };

        let pipeline_key = key;
        let device = self.device.clone();
        let shader_for_pipeline = shader.clone();
        let pipeline_result = entry.pipeline.get_or_init(move || {
            let label = format!(
                "st.tensor.wgpu_dense.matmul_pipeline.{:?}.tile{}x{}x{}",
                pipeline_key.dtype, pipeline_key.tile_m, pipeline_key.tile_n, pipeline_key.tile_k
            );
            create_compute_pipeline(
                device.as_ref(),
                &label,
                Some(layout),
                shader_for_pipeline.as_ref(),
                "main",
            )
            .map(Arc::new)
            .map_err(|err| err.to_string())
        });
        match pipeline_result {
            Ok(pipeline) => Ok(Arc::clone(pipeline)),
            Err(err) => Err(anyhow!(err.clone())),
        }
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

fn pack_rhs_f32(packed: &PackedB, _tile: TileConfig) -> (Vec<f32>, usize) {
    let inner = packed.inner();
    let cols = packed.cols();
    let mut row_major = vec![0.0f32; inner * cols];
    let slice = packed.as_slice();
    let layout = packed.layout();

    for k in 0..inner {
        for col in 0..cols {
            let value = match layout {
                PackedLayout::ColMajor => slice[col * inner + k],
                PackedLayout::Tiled { .. } => slice[col * inner + k],
            };
            row_major[k * cols + col] = value;
        }
    }

    let total_elems = row_major.len();
    (row_major, total_elems)
}

fn pack_rhs_int8(packed: &PackedB, _tile: TileConfig) -> (Vec<i32>, Vec<f32>, usize) {
    let inner = packed.inner();
    let cols = packed.cols();
    let stride = inner.div_ceil(4);
    let mut quantized = vec![0i32; cols * stride];
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

    for col in 0..cols {
        let inv_scale = 1.0 / scales[col];
        for pack_idx in 0..stride {
            let mut word = 0u32;
            for lane in 0..4 {
                let k = pack_idx * 4 + lane;
                let quant = if k >= inner {
                    0i32
                } else {
                    let raw = match layout {
                        PackedLayout::ColMajor => slice[col * inner + k],
                        PackedLayout::Tiled { .. } => slice[col * inner + k],
                    };
                    (raw * inv_scale).round().clamp(-127.0, 127.0) as i32
                };
                let byte = (quant as i8) as u8 as u32;
                word |= byte << (lane * 8);
            }
            quantized[col * stride + pack_idx] = word as i32;
        }
    }

    let total_elems = cols * stride * 4;
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
    autotune_cache: Mutex<HashMap<String, TileConfig>>,
    bind_layout: Arc<BindGroupLayout>,
    pipeline_layout: Arc<PipelineLayout>,
    zero_storage: OnceLock<Arc<Buffer>>,
    zero_scales: OnceLock<Arc<Buffer>>,
    shader_f16: bool,
    supports_subgroup: bool,
    adapter_info: AdapterInfo,
    softmax_layout: BindGroupLayout,
    softmax_workgroup_pipeline: Option<Arc<ComputePipeline>>,
    softmax_subgroup_pipeline: Option<Arc<ComputePipeline>>,
    softmax_zspace_layout: Option<BindGroupLayout>,
    softmax_zspace_pipeline: Option<Arc<ComputePipeline>>,
    softmax_spiral_layout: Option<BindGroupLayout>,
    softmax_spiral_pipeline: Option<Arc<ComputePipeline>>,
    softmax_variants: Mutex<HashMap<String, SoftmaxVariant>>,
    softmax_history: Mutex<Vec<SoftmaxSelectionRecord>>,
    softmax_telemetry_keys: Mutex<HashMap<String, AutotuneKey>>,
    fused_attention: Option<FusedAttentionKernel>,
    fused_gelu_back_layout: BindGroupLayout,
    fused_gelu_back_pipeline: Arc<ComputePipeline>,
    reduce_db_layout: BindGroupLayout,
    reduce_db_pipeline: Arc<ComputePipeline>,
    layer_norm_layout: BindGroupLayout,
    layer_norm_pipeline: Arc<ComputePipeline>,
    tensor_util_layout: BindGroupLayout,
    tensor_util_scale_pipeline: Arc<ComputePipeline>,
    tensor_util_add_pipeline: Arc<ComputePipeline>,
    tensor_util_hadamard_pipeline: Arc<ComputePipeline>,
    tensor_util_mul_row_pipeline: Arc<ComputePipeline>,
    tensor_util_row_affine_pipeline: Arc<ComputePipeline>,
    tensor_util_dynamic_klein_gordon_forward_pipeline: Arc<ComputePipeline>,
    tensor_util_dynamic_klein_gordon_backward_pipeline: Arc<ComputePipeline>,
    tensor_util_dynamic_hamilton_jacobi_forward_pipeline: Arc<ComputePipeline>,
    tensor_util_dynamic_hamilton_jacobi_backward_pipeline: Arc<ComputePipeline>,
    tensor_util_dynamic_schrodinger_forward_pipeline: Arc<ComputePipeline>,
    tensor_util_dynamic_schrodinger_backward_pipeline: Arc<ComputePipeline>,
    tensor_util_sequence_last_step_gather_pipeline: Arc<ComputePipeline>,
    tensor_util_sequence_last_step_scatter_pipeline: Arc<ComputePipeline>,
    tensor_util_zspace_coherence_scan_forward_pipeline: Arc<ComputePipeline>,
    tensor_util_zspace_coherence_scan_backward_pipeline: Arc<ComputePipeline>,
    tensor_util_add_scaled_pipeline: Arc<ComputePipeline>,
    tensor_util_sub_pipeline: Arc<ComputePipeline>,
    tensor_util_transpose_pipeline: Arc<ComputePipeline>,
    tensor_util_embedding_gather_pipeline: Arc<ComputePipeline>,
    tensor_util_embedding_scatter_add_pipeline: Arc<ComputePipeline>,
    tensor_util_hypergrad_accumulate_wave_pipeline: Arc<ComputePipeline>,
    tensor_util_relu_pipeline: Arc<ComputePipeline>,
    tensor_util_relu_backward_pipeline: Arc<ComputePipeline>,
    tensor_util_lstm_forward_gate_step_pipeline: Arc<ComputePipeline>,
    tensor_util_max_pool2d_forward_pipeline: Arc<ComputePipeline>,
    tensor_util_max_pool2d_backward_pipeline: Arc<ComputePipeline>,
    tensor_util_avg_pool2d_forward_pipeline: Arc<ComputePipeline>,
    tensor_util_avg_pool2d_backward_pipeline: Arc<ComputePipeline>,
    tensor_util_sum_squares_pipeline: Arc<ComputePipeline>,
    tensor_util_sum_abs_pipeline: Arc<ComputePipeline>,
    tensor_util_mse_loss_forward_pipeline: Arc<ComputePipeline>,
    tensor_util_mse_loss_backward_pipeline: Arc<ComputePipeline>,
    tensor_util_categorical_cross_entropy_forward_pipeline: Arc<ComputePipeline>,
    tensor_util_categorical_cross_entropy_backward_pipeline: Arc<ComputePipeline>,
    tensor_util_hyperbolic_cross_entropy_forward_pipeline: Arc<ComputePipeline>,
    tensor_util_hyperbolic_cross_entropy_backward_pipeline: Arc<ComputePipeline>,
    tensor_util_zspace_softmax_backward_fixed_pipeline: Arc<ComputePipeline>,
    tensor_util_add_row_pipeline: Arc<ComputePipeline>,
    tensor_util_sum_axis0_pipeline: Arc<ComputePipeline>,
    tensor_util_sum_axis0_scaled_pipeline: Arc<ComputePipeline>,
    tensor_util_project_to_poincare_pipeline: Arc<ComputePipeline>,
    tensor_util_wave_gate_project_pipeline: Arc<ComputePipeline>,
    tensor_util_wave_gate_backward_pipeline: Arc<ComputePipeline>,
    fused_conv_layout: BindGroupLayout,
    fused_conv_pipeline_layout: PipelineLayout,
    fused_conv_pipelines: Mutex<HashMap<TileConfig, Arc<ComputePipeline>>>,
    fused_grad_input_layout: BindGroupLayout,
    fused_grad_input_pipeline_layout: PipelineLayout,
    fused_grad_input_pipeline: OnceLock<Arc<ComputePipeline>>,
    ramanujan_layout: BindGroupLayout,
    ramanujan_pipeline_layout: PipelineLayout,
    ramanujan_pipeline: OnceLock<Arc<ComputePipeline>>,
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
            let opts = wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            };
            if let Some(adapter) = instance.request_adapter(&opts).await {
                return Some(adapter);
            }

            let opts = wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: false,
            };
            if let Some(adapter) = instance.request_adapter(&opts).await {
                return Some(adapter);
            }

            let opts = wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            };
            instance.request_adapter(&opts).await
        })
        .ok_or_else(|| "no suitable WGPU adapter".to_string())?;

        let adapter_info = adapter.get_info();
        let adapter_features = adapter.features();
        let mut requested_features = wgpu::Features::empty();
        let want_f16 = cfg!(feature = "wgpu_f16");
        let shader_f16 = want_f16 && adapter_features.contains(wgpu::Features::SHADER_F16);
        if shader_f16 {
            requested_features |= wgpu::Features::SHADER_F16;
        }
        let supports_subgroup = adapter_features.contains(wgpu::Features::SUBGROUP);
        if supports_subgroup {
            requested_features |= wgpu::Features::SUBGROUP;
        }

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
            ],
        });

        let softmax_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.softmax.pipeline_layout"),
                bind_group_layouts: &[&softmax_layout],
                push_constant_ranges: &[],
            });

        let softmax_workgroup_pipeline = create_wgsl_module(
            device.as_ref(),
            "st.tensor.wgpu_dense.softmax",
            ROW_SOFTMAX_WGSL,
        )
        .and_then(|shader| {
            create_compute_pipeline(
                device.as_ref(),
                "st.tensor.wgpu_dense.softmax",
                Some(&softmax_pipeline_layout),
                &shader,
                "main_cs",
            )
            .map(Arc::new)
        })
        .ok();

        let softmax_subgroup_pipeline = if supports_subgroup {
            create_wgsl_module(
                device.as_ref(),
                "st.tensor.wgpu_dense.softmax.subgroup",
                ROW_SOFTMAX_SUBGROUP_WGSL,
            )
            .and_then(|shader| {
                create_compute_pipeline(
                    device.as_ref(),
                    "st.tensor.wgpu_dense.softmax.subgroup",
                    Some(&softmax_pipeline_layout),
                    &shader,
                    "main_cs",
                )
                .map(Arc::new)
            })
            .ok()
        } else {
            None
        };

        let softmax_zspace_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.softmax_zspace.layout"),
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

        let softmax_zspace_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.softmax_zspace.pipeline_layout"),
                bind_group_layouts: &[&softmax_zspace_layout],
                push_constant_ranges: &[],
            });

        let softmax_zspace_pipeline = create_wgsl_module(
            device.as_ref(),
            "st.tensor.wgpu_dense.softmax_zspace",
            SOFTMAX_ZSPACE_WGSL,
        )
        .and_then(|shader| {
            create_compute_pipeline(
                device.as_ref(),
                "st.tensor.wgpu_dense.softmax_zspace",
                Some(&softmax_zspace_pipeline_layout),
                &shader,
                "main",
            )
            .map(Arc::new)
        })
        .ok();

        let softmax_spiral_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.softmax_spiral.layout"),
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

        let softmax_spiral_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.softmax_spiral.pipeline_layout"),
                bind_group_layouts: &[&softmax_spiral_layout],
                push_constant_ranges: &[],
            });

        let softmax_spiral_pipeline = create_wgsl_module(
            device.as_ref(),
            "st.tensor.wgpu_dense.softmax_spiral",
            SOFTMAX_SPIRAL_WGSL,
        )
        .and_then(|shader| {
            create_compute_pipeline(
                device.as_ref(),
                "st.tensor.wgpu_dense.softmax_spiral",
                Some(&softmax_spiral_pipeline_layout),
                &shader,
                "main",
            )
            .map(Arc::new)
        })
        .ok();

        let fused_gelu_back_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_gelu_back_layout"),
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

        let fused_gelu_back_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_gelu_back.pipeline_layout"),
                bind_group_layouts: &[&fused_gelu_back_layout],
                push_constant_ranges: &[],
            });

        let fused_gelu_back_shader_source = instantiate_fused_gelu_back_template(
            FUSED_GELU_BACK_WGSL_TEMPLATE,
            FUSED_GELU_BACK_WG_ROWS,
            FUSED_GELU_BACK_WG_COLS,
        );
        let fused_gelu_back_shader = create_wgsl_module(
            device.as_ref(),
            "st.tensor.wgpu_dense.fused_gelu_back",
            fused_gelu_back_shader_source.as_str(),
        )
        .map_err(|err| err.to_string())?;
        let fused_gelu_back_pipeline = Arc::new(
            create_compute_pipeline(
                device.as_ref(),
                "st.tensor.wgpu_dense.fused_gelu_back",
                Some(&fused_gelu_back_pipeline_layout),
                &fused_gelu_back_shader,
                "main",
            )
            .map_err(|err| err.to_string())?,
        );

        let reduce_db_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("st.tensor.wgpu_dense.reduce_db_layout"),
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

        let reduce_db_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.reduce_db.pipeline_layout"),
                bind_group_layouts: &[&reduce_db_layout],
                push_constant_ranges: &[],
            });

        let reduce_db_shader_source = instantiate_reduce_db_template(
            REDUCE_DB_WGSL_TEMPLATE,
            FUSED_GELU_BACK_WG_COLS,
            REDUCE_DB_WORKGROUP,
        );
        let reduce_db_shader = create_wgsl_module(
            device.as_ref(),
            "st.tensor.wgpu_dense.reduce_db",
            reduce_db_shader_source.as_str(),
        )
        .map_err(|err| err.to_string())?;
        let reduce_db_pipeline = Arc::new(
            create_compute_pipeline(
                device.as_ref(),
                "st.tensor.wgpu_dense.reduce_db",
                Some(&reduce_db_pipeline_layout),
                &reduce_db_shader,
                "reduce",
            )
            .map_err(|err| err.to_string())?,
        );

        let layer_norm_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("st.tensor.wgpu_dense.layer_norm.layout"),
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

        let layer_norm_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.layer_norm.pipeline_layout"),
                bind_group_layouts: &[&layer_norm_layout],
                push_constant_ranges: &[],
            });

        let layer_norm_shader = create_wgsl_module(
            device.as_ref(),
            "st.tensor.wgpu_dense.layer_norm",
            LAYER_NORM_WGSL,
        )
        .map_err(|err| err.to_string())?;
        let layer_norm_pipeline = Arc::new(
            create_compute_pipeline(
                device.as_ref(),
                "st.tensor.wgpu_dense.layer_norm",
                Some(&layer_norm_pipeline_layout),
                &layer_norm_shader,
                "main",
            )
            .map_err(|err| err.to_string())?,
        );

        let tensor_util_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.tensor_util.layout"),
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
        let tensor_util_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.tensor_util.pipeline_layout"),
                bind_group_layouts: &[&tensor_util_layout],
                push_constant_ranges: &[],
            });
        let tensor_util_shader = create_wgsl_module(
            device.as_ref(),
            "st.tensor.wgpu_dense.tensor_util",
            TENSOR_UTILS_WGSL,
        )
        .map_err(|err| err.to_string())?;
        let tensor_util_pipeline = |entry_point: &str| -> Result<Arc<ComputePipeline>, String> {
            create_compute_pipeline(
                device.as_ref(),
                &format!("st.tensor.wgpu_dense.tensor_util.{entry_point}"),
                Some(&tensor_util_pipeline_layout),
                &tensor_util_shader,
                entry_point,
            )
            .map(Arc::new)
            .map_err(|err| err.to_string())
        };
        let tensor_util_scale_pipeline = tensor_util_pipeline("scale")?;
        let tensor_util_add_pipeline = tensor_util_pipeline("add")?;
        let tensor_util_hadamard_pipeline = tensor_util_pipeline("hadamard")?;
        let tensor_util_mul_row_pipeline = tensor_util_pipeline("mul_row")?;
        let tensor_util_row_affine_pipeline = tensor_util_pipeline("row_affine")?;
        let tensor_util_dynamic_klein_gordon_forward_pipeline =
            tensor_util_pipeline("dynamic_klein_gordon_forward")?;
        let tensor_util_dynamic_klein_gordon_backward_pipeline =
            tensor_util_pipeline("dynamic_klein_gordon_backward")?;
        let tensor_util_dynamic_hamilton_jacobi_forward_pipeline =
            tensor_util_pipeline("dynamic_hamilton_jacobi_forward")?;
        let tensor_util_dynamic_hamilton_jacobi_backward_pipeline =
            tensor_util_pipeline("dynamic_hamilton_jacobi_backward")?;
        let tensor_util_dynamic_schrodinger_forward_pipeline =
            tensor_util_pipeline("dynamic_schrodinger_forward")?;
        let tensor_util_dynamic_schrodinger_backward_pipeline =
            tensor_util_pipeline("dynamic_schrodinger_backward")?;
        let tensor_util_sequence_last_step_gather_pipeline =
            tensor_util_pipeline("sequence_last_step_gather")?;
        let tensor_util_sequence_last_step_scatter_pipeline =
            tensor_util_pipeline("sequence_last_step_scatter")?;
        let tensor_util_zspace_coherence_scan_forward_pipeline =
            tensor_util_pipeline("zspace_coherence_scan_forward")?;
        let tensor_util_zspace_coherence_scan_backward_pipeline =
            tensor_util_pipeline("zspace_coherence_scan_backward")?;
        let tensor_util_add_scaled_pipeline = tensor_util_pipeline("add_scaled")?;
        let tensor_util_sub_pipeline = tensor_util_pipeline("sub")?;
        let tensor_util_transpose_pipeline = tensor_util_pipeline("transpose")?;
        let tensor_util_embedding_gather_pipeline = tensor_util_pipeline("embedding_gather")?;
        let tensor_util_embedding_scatter_add_pipeline =
            tensor_util_pipeline("embedding_scatter_add")?;
        let tensor_util_hypergrad_accumulate_wave_pipeline =
            tensor_util_pipeline("hypergrad_accumulate_wave")?;
        let tensor_util_relu_pipeline = tensor_util_pipeline("relu")?;
        let tensor_util_relu_backward_pipeline = tensor_util_pipeline("relu_backward")?;
        let tensor_util_lstm_forward_gate_step_pipeline =
            tensor_util_pipeline("lstm_forward_gate_step")?;
        let tensor_util_max_pool2d_forward_pipeline = tensor_util_pipeline("max_pool2d_forward")?;
        let tensor_util_max_pool2d_backward_pipeline = tensor_util_pipeline("max_pool2d_backward")?;
        let tensor_util_avg_pool2d_forward_pipeline = tensor_util_pipeline("avg_pool2d_forward")?;
        let tensor_util_avg_pool2d_backward_pipeline = tensor_util_pipeline("avg_pool2d_backward")?;
        let tensor_util_sum_squares_pipeline = tensor_util_pipeline("sum_squares")?;
        let tensor_util_sum_abs_pipeline = tensor_util_pipeline("sum_abs")?;
        let tensor_util_mse_loss_forward_pipeline = tensor_util_pipeline("mse_loss_forward")?;
        let tensor_util_mse_loss_backward_pipeline = tensor_util_pipeline("mse_loss_backward")?;
        let tensor_util_categorical_cross_entropy_forward_pipeline =
            tensor_util_pipeline("categorical_cross_entropy_forward")?;
        let tensor_util_categorical_cross_entropy_backward_pipeline =
            tensor_util_pipeline("categorical_cross_entropy_backward")?;
        let tensor_util_hyperbolic_cross_entropy_forward_pipeline =
            tensor_util_pipeline("hyperbolic_cross_entropy_forward")?;
        let tensor_util_hyperbolic_cross_entropy_backward_pipeline =
            tensor_util_pipeline("hyperbolic_cross_entropy_backward")?;
        let tensor_util_zspace_softmax_backward_fixed_pipeline =
            tensor_util_pipeline("zspace_softmax_backward_fixed")?;
        let tensor_util_add_row_pipeline = tensor_util_pipeline("add_row")?;
        let tensor_util_sum_axis0_pipeline = tensor_util_pipeline("sum_axis0")?;
        let tensor_util_sum_axis0_scaled_pipeline = tensor_util_pipeline("sum_axis0_scaled")?;
        let tensor_util_project_to_poincare_pipeline = tensor_util_pipeline("project_to_poincare")?;
        let tensor_util_wave_gate_project_pipeline = tensor_util_pipeline("wave_gate_project")?;
        let tensor_util_wave_gate_backward_pipeline = tensor_util_pipeline("wave_gate_backward")?;

        let fused_attention = {
            let shader_source = FUSED_ATTENTION_WGSL_TEMPLATE
                .replace("{WORKGROUP_SIZE}", &FUSED_ATTENTION_WORKGROUP.to_string())
                .replace("{MAX_HEAD_DIM}", &FUSED_ATTENTION_MAX_HEAD_DIM.to_string());
            create_wgsl_module(
                device.as_ref(),
                "st.tensor.wgpu_dense.fused_attention.shader",
                shader_source.as_str(),
            )
            .and_then(|shader| {
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
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("st.tensor.wgpu_dense.fused_attention.pipeline_layout"),
                        bind_group_layouts: &[&layout],
                        push_constant_ranges: &[],
                    });
                create_compute_pipeline(
                    device.as_ref(),
                    "st.tensor.wgpu_dense.fused_attention",
                    Some(&pipeline_layout),
                    &shader,
                    "main",
                )
                .map(Arc::new)
                .map(|pipeline| FusedAttentionKernel {
                    layout,
                    pipeline,
                    max_head_dim: FUSED_ATTENTION_MAX_HEAD_DIM,
                })
            })
            .ok()
        };

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

        let fused_grad_input_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_grad_input.layout"),
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
        let fused_grad_input_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.fused_grad_input.pipeline_layout"),
                bind_group_layouts: &[&fused_grad_input_layout],
                push_constant_ranges: &[],
            });

        let ramanujan_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("st.tensor.wgpu_dense.ramanujan.layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
        let ramanujan_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("st.tensor.wgpu_dense.ramanujan.pipeline_layout"),
                bind_group_layouts: &[&ramanujan_layout],
                push_constant_ranges: &[],
            });

        Ok(Self {
            context: WgpuContext::new(device.clone(), queue.clone()),
            pipeline_cache: PipelineCache::new(device.clone()),
            weights_cache: Mutex::new(HashMap::new()),
            autotune_cache: Mutex::new(HashMap::new()),
            bind_layout,
            pipeline_layout,
            zero_storage: OnceLock::new(),
            zero_scales: OnceLock::new(),
            shader_f16,
            supports_subgroup,
            adapter_info,
            softmax_layout,
            softmax_workgroup_pipeline,
            softmax_subgroup_pipeline,
            softmax_variants: Mutex::new(HashMap::new()),
            softmax_history: Mutex::new(Vec::new()),
            softmax_telemetry_keys: Mutex::new(HashMap::new()),
            fused_attention,
            fused_gelu_back_layout,
            fused_gelu_back_pipeline,
            reduce_db_layout,
            reduce_db_pipeline,
            layer_norm_layout,
            layer_norm_pipeline,
            tensor_util_layout,
            tensor_util_scale_pipeline,
            tensor_util_add_pipeline,
            tensor_util_hadamard_pipeline,
            tensor_util_mul_row_pipeline,
            tensor_util_row_affine_pipeline,
            tensor_util_dynamic_klein_gordon_forward_pipeline,
            tensor_util_dynamic_klein_gordon_backward_pipeline,
            tensor_util_dynamic_hamilton_jacobi_forward_pipeline,
            tensor_util_dynamic_hamilton_jacobi_backward_pipeline,
            tensor_util_dynamic_schrodinger_forward_pipeline,
            tensor_util_dynamic_schrodinger_backward_pipeline,
            tensor_util_sequence_last_step_gather_pipeline,
            tensor_util_sequence_last_step_scatter_pipeline,
            tensor_util_zspace_coherence_scan_forward_pipeline,
            tensor_util_zspace_coherence_scan_backward_pipeline,
            tensor_util_add_scaled_pipeline,
            tensor_util_sub_pipeline,
            tensor_util_transpose_pipeline,
            tensor_util_embedding_gather_pipeline,
            tensor_util_embedding_scatter_add_pipeline,
            tensor_util_hypergrad_accumulate_wave_pipeline,
            tensor_util_relu_pipeline,
            tensor_util_relu_backward_pipeline,
            tensor_util_lstm_forward_gate_step_pipeline,
            tensor_util_max_pool2d_forward_pipeline,
            tensor_util_max_pool2d_backward_pipeline,
            tensor_util_avg_pool2d_forward_pipeline,
            tensor_util_avg_pool2d_backward_pipeline,
            tensor_util_sum_squares_pipeline,
            tensor_util_sum_abs_pipeline,
            tensor_util_mse_loss_forward_pipeline,
            tensor_util_mse_loss_backward_pipeline,
            tensor_util_categorical_cross_entropy_forward_pipeline,
            tensor_util_categorical_cross_entropy_backward_pipeline,
            tensor_util_hyperbolic_cross_entropy_forward_pipeline,
            tensor_util_hyperbolic_cross_entropy_backward_pipeline,
            tensor_util_zspace_softmax_backward_fixed_pipeline,
            tensor_util_add_row_pipeline,
            tensor_util_sum_axis0_pipeline,
            tensor_util_sum_axis0_scaled_pipeline,
            tensor_util_project_to_poincare_pipeline,
            tensor_util_wave_gate_project_pipeline,
            tensor_util_wave_gate_backward_pipeline,
            fused_conv_layout,
            fused_conv_pipeline_layout,
            fused_conv_pipelines: Mutex::new(HashMap::new()),
            fused_grad_input_layout,
            fused_grad_input_pipeline_layout,
            fused_grad_input_pipeline: OnceLock::new(),
            ramanujan_layout,
            ramanujan_pipeline_layout,
            ramanujan_pipeline: OnceLock::new(),
            softmax_zspace_layout: softmax_zspace_pipeline
                .as_ref()
                .map(|_| softmax_zspace_layout),
            softmax_zspace_pipeline,
            softmax_spiral_layout: softmax_spiral_pipeline
                .as_ref()
                .map(|_| softmax_spiral_layout),
            softmax_spiral_pipeline,
        })
    }

    fn device(&self) -> &Device {
        self.context.device()
    }

    fn queue(&self) -> &Queue {
        self.context.queue()
    }

    fn softmax_pipeline_variant(&self, variant: SoftmaxVariant) -> Option<Arc<ComputePipeline>> {
        match variant {
            SoftmaxVariant::Workgroup => self.softmax_workgroup_pipeline.as_ref().map(Arc::clone),
            SoftmaxVariant::Subgroup => self.softmax_subgroup_pipeline.as_ref().map(Arc::clone),
        }
    }

    fn softmax_registry_key(
        &self,
        rows: usize,
        cols: usize,
        layout: &SoftmaxLayoutDesc,
    ) -> AutotuneKey {
        let limits = self.device().limits();
        let subgroup_size = if self.supports_subgroup {
            limits
                .max_compute_invocations_per_workgroup
                .min(SOFTMAX_WORKGROUP_SIZE as u32)
                .max(1)
        } else {
            1
        };
        let shared_kb = (limits.max_compute_workgroup_storage_size / 1024).max(1);
        let mut driver = self.adapter_info.driver.clone();
        if !self.adapter_info.driver_info.is_empty() {
            if driver.is_empty() {
                driver = self.adapter_info.driver_info.clone();
            } else {
                driver = format!("{driver} {}", self.adapter_info.driver_info);
            }
        }
        let device_profile = DeviceProfile::new(
            self.adapter_info.name.clone(),
            self.adapter_info.device,
            subgroup_size,
            shared_kb,
            driver,
        );
        let signature = format!(
            "wgpu.softmax|{}x{}|flags{}|tile{}|stripes{}",
            rows, cols, layout.flags, layout.chimera_tile, layout.chimera_stripes
        );
        let kernel_profile = KernelProfile::new(SOFTMAX_AUTOTUNE_REVISION, signature);
        AutotuneKey::new(device_profile, kernel_profile)
    }

    fn record_softmax_telemetry(
        &self,
        cache_key: &str,
        bucket_rows: usize,
        bucket_cols: usize,
        layout: &SoftmaxLayoutDesc,
        sample: TelemetrySample,
    ) {
        let registry_key = self.softmax_registry_key(bucket_rows, bucket_cols, layout);
        if let Ok(mut map) = self.softmax_telemetry_keys.lock() {
            map.insert(cache_key.to_string(), registry_key.clone());
        }
        if let Ok(mut registry) = global_autotune_registry().lock() {
            registry.record(registry_key, sample);
        }
    }

    fn select_softmax_pipeline(
        &self,
        rows: usize,
        cols: usize,
        layout: &SoftmaxLayoutDesc,
    ) -> Result<(Arc<ComputePipeline>, SoftmaxVariant), String> {
        let workgroup_pipeline = self
            .softmax_pipeline_variant(SoftmaxVariant::Workgroup)
            .ok_or_else(|| "row softmax pipeline unavailable".to_string())?;

        if !self.supports_subgroup {
            return Ok((workgroup_pipeline, SoftmaxVariant::Workgroup));
        }

        if self
            .softmax_pipeline_variant(SoftmaxVariant::Subgroup)
            .is_none()
        {
            return Ok((workgroup_pipeline, SoftmaxVariant::Workgroup));
        }

        let bucket_rows = quantize_dimension(rows);
        let bucket_cols = quantize_dimension(cols);
        let key = softmax_cache_key(self, bucket_rows, bucket_cols, layout);
        let store_path = autotune_store_path();

        if let Ok(cache) = self.softmax_variants.lock() {
            if let Some(&variant) = cache.get(&key) {
                if let Some(pipeline) = self.softmax_pipeline_variant(variant) {
                    return Ok((pipeline, variant));
                }
            }
        }

        let context = SoftmaxAutoContext {
            rows: bucket_rows,
            cols: bucket_cols,
            layout_flags: layout.flags,
            chimera_tile: layout.chimera_tile,
            chimera_stripes: layout.chimera_stripes,
            has_subgroup: true,
        };

        if let Some(path) = store_path.as_ref() {
            if let Some(stored) =
                load_best_typed(path.as_path(), &key, &context, None::<StoredSoftmaxVariant>)
            {
                if let Some(stored_variant) = SoftmaxVariant::from_str(&stored.variant) {
                    if let Some(pipeline) = self.softmax_pipeline_variant(stored_variant) {
                        if let Ok(mut cache) = self.softmax_variants.lock() {
                            cache.insert(key.clone(), stored_variant);
                        }
                        return Ok((pipeline, stored_variant));
                    }
                }
            }
        }

        let autotune_enabled = autotune_env_enabled() && store_path.is_some();
        type SoftmaxAutotuneBest = (
            SoftmaxVariant,
            f64,
            Option<SoftmaxZProjectMetrics>,
            Option<SoftmaxBayesEvidence>,
            Option<SoftmaxMetropolisEvidence>,
            Option<SoftmaxSpiralAnnealEvidence>,
            Option<SoftmaxSpiralConsensusEvidence>,
            f64,
        );
        let mut best: Option<SoftmaxAutotuneBest> = None;
        for &candidate in &[SoftmaxVariant::Workgroup, SoftmaxVariant::Subgroup] {
            let Some(pipeline) = self.softmax_pipeline_variant(candidate) else {
                continue;
            };
            match self.microbenchmark_softmax(pipeline.as_ref(), rows, cols, layout) {
                Ok((score_s, projection)) => {
                    let score_ms = score_s * 1_000.0;
                    let bayes = self.bayesian_refine_softmax_score(
                        candidate,
                        score_ms,
                        projection.as_ref(),
                    );
                    let metropolis =
                        self.metropolis_multi_try_softmax(candidate, score_ms, projection.as_ref());
                    let anneal = self.spiral_anneal_softmax(
                        candidate,
                        score_ms,
                        projection.as_ref(),
                        bayes.as_ref(),
                        metropolis.as_ref(),
                    );
                    let consensus = self.spiral_consensus_softmax(
                        candidate,
                        score_ms,
                        projection.as_ref(),
                        bayes.as_ref(),
                        metropolis.as_ref(),
                        anneal.as_ref(),
                    );
                    let mut effective_ms = score_ms;
                    if let Some(ref evidence) = bayes {
                        effective_ms = effective_ms.min(evidence.posterior_ms);
                    }
                    if let Some(ref mtm) = metropolis {
                        effective_ms = effective_ms.min(mtm.expected_ms);
                    }
                    if let Some(ref anneal) = anneal {
                        effective_ms = effective_ms.min(anneal.annealed_ms);
                    }
                    if let Some(ref consensus) = consensus {
                        effective_ms = effective_ms.min(consensus.consensus_ms);
                    }
                    let update = best
                        .map(|(_, _, _, _, _, _, _, best_ms)| effective_ms < best_ms)
                        .unwrap_or(true);
                    if update {
                        best = Some((
                            candidate,
                            score_s,
                            projection,
                            bayes,
                            metropolis,
                            anneal,
                            consensus,
                            effective_ms,
                        ));
                    }
                }
                Err(_) => continue,
            }
        }

        let measured = best.is_some();
        let (variant, score_s, projection, bayes, metropolis, anneal, consensus, _) = best
            .unwrap_or((
                SoftmaxVariant::Workgroup,
                0.0,
                None,
                None,
                None,
                None,
                None,
                f64::MAX,
            ));
        let pipeline = self
            .softmax_pipeline_variant(variant)
            .ok_or_else(|| "row softmax pipeline unavailable".to_string())?;

        if let Ok(mut cache) = self.softmax_variants.lock() {
            cache.insert(key.clone(), variant);
        }

        if measured {
            if let Some(sample) = make_softmax_telemetry_sample(rows, cols, score_s) {
                self.record_softmax_telemetry(&key, bucket_rows, bucket_cols, layout, sample);
            }
            if let Ok(mut history) = self.softmax_history.lock() {
                history.push(SoftmaxSelectionRecord {
                    key: key.clone(),
                    variant,
                    score_ms: score_s * 1_000.0,
                    samples: SOFTMAX_AUTOTUNE_SAMPLES,
                    zmetrics: projection,
                    bayes,
                    metropolis,
                    anneal,
                    consensus,
                });
                let len = history.len();
                if len > SOFTMAX_HISTORY_LIMIT {
                    let remove = len - SOFTMAX_HISTORY_LIMIT;
                    history.drain(0..remove);
                }
            }

            if let (true, Some(path)) = (autotune_enabled, store_path.as_ref()) {
                let stored = StoredSoftmaxVariant {
                    variant: variant.as_str().to_string(),
                };
                let _ = record_best(path.as_path(), &key, &context, score_s, &stored);
            }
        }

        Ok((pipeline, variant))
    }

    fn microbenchmark_softmax(
        &self,
        pipeline: &ComputePipeline,
        rows: usize,
        cols: usize,
        layout: &SoftmaxLayoutDesc,
    ) -> Result<(f64, Option<SoftmaxZProjectMetrics>), String> {
        if rows == 0 || cols == 0 {
            return Err("rows and cols must be positive".into());
        }

        let rows_u32 = u32::try_from(rows).map_err(|_| "rows exceed u32::MAX".to_string())?;
        let cols_u32 = u32::try_from(cols).map_err(|_| "cols exceed u32::MAX".to_string())?;

        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| "matrix dimensions overflow".to_string())?;

        let device = self.device();
        let queue = self.queue();

        let input_data = vec![0.0f32; total];
        let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax.autotune.input"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buf = allocate_output(
            device,
            "st.tensor.wgpu_dense.softmax.autotune.output",
            total,
        );
        let mask_buf = if (layout.flags & SOFTMAX_FLAG_HARDMAX_MASK) != 0 {
            Some(allocate_output(
                device,
                "st.tensor.wgpu_dense.softmax.autotune.mask",
                total,
            ))
        } else {
            None
        };

        let params = RowSoftmaxParams {
            rows: rows_u32,
            cols: cols_u32,
            in_stride: layout.in_stride,
            out_stride: layout.out_stride,
            chimera_tile: layout.chimera_tile,
            chimera_stripes: layout.chimera_stripes,
            flags: layout.flags,
            mask_stride: layout.out_stride,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax.autotune.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let mask_binding = mask_buf
            .as_ref()
            .map(|buffer| buffer.as_entire_binding())
            .unwrap_or_else(|| output_buf.as_entire_binding());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax.autotune.bg"),
            layout: &self.softmax_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mask_binding,
                },
            ],
        });

        for _ in 0..SOFTMAX_AUTOTUNE_WARMUP {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("st.tensor.wgpu_dense.softmax.autotune.warmup"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("st.tensor.wgpu_dense.softmax.autotune.warmup.pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(rows_u32, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
        }

        let mut total = 0.0f64;
        for _ in 0..SOFTMAX_AUTOTUNE_SAMPLES {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("st.tensor.wgpu_dense.softmax.autotune"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("st.tensor.wgpu_dense.softmax.autotune.pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(rows_u32, 1, 1);
            }
            let start = Instant::now();
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::Maintain::Wait);
            total += start.elapsed().as_secs_f64();
        }

        let avg = total / SOFTMAX_AUTOTUNE_SAMPLES as f64;
        let projection = self.project_softmax_zspace(rows, cols, layout, &output_buf);
        Ok((avg, projection))
    }

    fn adapter_info(&self) -> &AdapterInfo {
        &self.adapter_info
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
        self.softmax_workgroup_pipeline.is_some()
    }

    fn select_tile_config(&self, rows: usize, inner: usize, cols: usize) -> TileConfig {
        autotune_tile_config(self, rows, inner, cols)
            .unwrap_or_else(|| fallback_tile_config(rows, inner, cols))
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
            WeightDType::Int8
        } else {
            WeightDType::F32
        };
        let key = RhsCacheKey::new(packed, tile, dtype.to_scalar());
        if let Some(existing) = self
            .weights_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
            .and_then(|weak| weak.upgrade())
        {
            return Ok(existing);
        }

        let (buffer, scales) = match dtype {
            WeightDType::Int8 => {
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
            WeightDType::F32 => {
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
            dtype: dtype.to_scalar(),
            buffer,
            scales,
            _cols: packed.cols(),
            _inner: packed.inner(),
        });

        self.weights_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(key, Arc::downgrade(&prepared));
        Ok(prepared)
    }

    fn fused_attention_kernel(&self) -> Option<&FusedAttentionKernel> {
        self.fused_attention.as_ref()
    }

    #[allow(clippy::too_many_arguments)]
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

    fn softmax_bind_group(
        &self,
        input: &Buffer,
        output: &Buffer,
        mask: Option<&Buffer>,
        params: &Buffer,
    ) -> BindGroup {
        let mask_binding = mask.unwrap_or(output);
        let descriptor = wgpu::BindGroupDescriptor {
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mask_binding.as_entire_binding(),
                },
            ],
        };
        self.device().create_bind_group(&descriptor)
    }

    fn softmax_zspace_bind_group(
        &self,
        output: &Buffer,
        metrics: &Buffer,
        params: &Buffer,
    ) -> Option<BindGroup> {
        let layout = self.softmax_zspace_layout.as_ref()?;
        let descriptor = wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_zspace.bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: metrics.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        };
        Some(self.device().create_bind_group(&descriptor))
    }

    fn softmax_spiral_bind_group(
        &self,
        softmax: &Buffer,
        mask: &Buffer,
        spiral: &Buffer,
        metrics: &Buffer,
        params: &Buffer,
    ) -> Option<BindGroup> {
        let layout = self.softmax_spiral_layout.as_ref()?;
        let descriptor = wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_spiral.bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: softmax.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mask.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spiral.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: metrics.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        };
        Some(self.device().create_bind_group(&descriptor))
    }

    fn project_softmax_zspace(
        &self,
        rows: usize,
        cols: usize,
        layout: &SoftmaxLayoutDesc,
        output: &Buffer,
    ) -> Option<SoftmaxZProjectMetrics> {
        let pipeline = self.softmax_zspace_pipeline.as_ref()?;
        if rows == 0 || cols == 0 {
            return None;
        }
        let rows_u32 = u32::try_from(rows).ok()?;
        let cols_u32 = u32::try_from(cols).ok()?;
        let metrics_len = rows.checked_mul(4)?;
        let metrics_size = (metrics_len * std::mem::size_of::<f32>()) as u64;
        let device = self.device();
        let queue = self.queue();

        let metrics_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_zspace.metrics"),
            size: metrics_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = SoftmaxZSpaceParams {
            rows: rows_u32,
            cols: cols_u32,
            stride: layout.out_stride,
            _pad: 0,
            golden_ratio: GOLDEN_RATIO,
            golden_angle: GOLDEN_ANGLE_RAD,
            min_energy: ZSPACE_MIN_ENERGY,
            _pad1: 0.0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_zspace.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.softmax_zspace_bind_group(output, &metrics_buf, &params_buf)?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_zspace.encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("st.tensor.wgpu_dense.softmax_zspace.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline.as_ref());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows_u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        let values = readback_f32(device, queue, &metrics_buf, metrics_len).ok()?;
        if values.len() != metrics_len {
            return None;
        }

        let rows_f32 = rows as f32;
        let inv_rows = rows_f32.recip();
        let mut sum_focus = 0.0;
        let mut sum_above = 0.0;
        let mut sum_here = 0.0;
        let mut sum_swirl = 0.0;
        for chunk in values.chunks_exact(4) {
            sum_focus += chunk[0].max(0.0);
            sum_above += chunk[1].clamp(0.0, 1.0);
            sum_here += chunk[2].clamp(0.0, 1.0);
            sum_swirl += chunk[3].clamp(-1.0, 1.0);
        }

        let mut focus = (sum_focus * inv_rows).clamp(0.0, 1.0);
        let mut above = (sum_above * inv_rows).clamp(0.0, 1.0);
        let mut here = (sum_here * inv_rows).clamp(0.0, 1.0);
        let mut beneath = (1.0 - (above + here)).clamp(0.0, 1.0);
        let total = above + here + beneath;
        if total > f32::EPSILON {
            let inv = total.recip();
            above *= inv;
            here *= inv;
            beneath *= inv;
        } else {
            above = 1.0 / 3.0;
            here = 1.0 / 3.0;
            beneath = 1.0 / 3.0;
        }

        focus = focus.clamp(0.0, 1.0);
        let swirl = (sum_swirl * inv_rows).clamp(-1.0, 1.0);
        let drift = (above - beneath).abs();
        let harmonic = (focus * (here + GOLDEN_RATIO.recip())).clamp(0.0, 1.0);
        let flux = ((drift + swirl.abs()) * harmonic).powf(1.0 / GOLDEN_RATIO);
        let geodesic =
            f64::from(focus.max(ZSPACE_MIN_ENERGY)) + f64::from(harmonic.max(ZSPACE_MIN_ENERGY));
        let sqrt_rank = (SOFTMAX_ZSPACE_LEECH_RANK.max(1) as f64).sqrt();
        let ramanujan_value = ramanujan_pi(SOFTMAX_ZSPACE_RAMANUJAN_ITERS).max(f64::EPSILON);
        let ramanujan_ratio_f64 = std::f64::consts::PI / ramanujan_value;
        let ramanujan_ratio = ramanujan_ratio_f64 as f32;
        let ramanujan_delta = (ramanujan_value - std::f64::consts::PI).abs() as f32;
        let ramanujan_iterations = SOFTMAX_ZSPACE_RAMANUJAN_ITERS as u32;
        let leech_raw = SOFTMAX_ZSPACE_LEECH_WEIGHT
            * LEECH_PACKING_DENSITY
            * geodesic
            * sqrt_rank
            * ramanujan_ratio_f64;
        let leech_enrichment = (leech_raw / SOFTMAX_ZSPACE_LEECH_SCALE).clamp(0.0, 1.0) as f32;
        Some(SoftmaxZProjectMetrics::new(
            focus,
            above,
            here,
            beneath,
            swirl,
            flux,
            leech_enrichment,
            ramanujan_ratio,
            ramanujan_delta,
            ramanujan_iterations,
        ))
    }

    fn prepare_spiral_consensus(
        &self,
        rows: usize,
        cols: usize,
        layout: &SoftmaxLayoutDesc,
        softmax: &Buffer,
        mask: &Buffer,
    ) -> Option<SpiralConsensusResources> {
        let pipeline = self.softmax_spiral_pipeline.as_ref()?;
        if rows == 0 || cols == 0 {
            return None;
        }

        let rows_u32 = u32::try_from(rows).ok()?;
        let cols_u32 = u32::try_from(cols).ok()?;
        let elements = rows.checked_mul(cols)?;
        let device = self.device();

        let spiral_buffer = allocate_output(
            device,
            "st.tensor.wgpu_dense.softmax_spiral.output",
            elements,
        );
        let metrics_len = rows.checked_mul(4)?;
        let metrics_size = (metrics_len * std::mem::size_of::<f32>()) as u64;
        let metrics_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_spiral.metrics"),
            size: metrics_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let approximation = ramanujan_pi(SPIRAL_PROJECTOR_RAMANUJAN_ITERS).max(f64::EPSILON);
        let pi = std::f64::consts::PI;
        let ramanujan_ratio = pi / approximation;
        let ramanujan_delta = (approximation - pi).abs();
        let sqrt_rank = (SPIRAL_PROJECTOR_RANK.max(1) as f64).sqrt();
        let leech_scale =
            SPIRAL_PROJECTOR_WEIGHT * SPIRAL_LEECH_PACKING_DENSITY * sqrt_rank * ramanujan_ratio;

        let params = SpiralConsensusParams {
            rows: rows_u32,
            cols: cols_u32,
            soft_stride: layout.out_stride,
            mask_stride: layout.out_stride,
            spiral_stride: layout.out_stride,
            chimera_tile: layout.chimera_tile,
            chimera_stripes: layout.chimera_stripes,
            flags: layout.flags,
            phi: GOLDEN_RATIO,
            phi_conjugate: GOLDEN_RATIO_CONJUGATE,
            phi_bias: GOLDEN_RATIO_BIAS,
            leech_scale: leech_scale as f32,
            ramanujan_ratio: ramanujan_ratio as f32,
            inv_cols: if cols_u32 == 0 {
                0.0
            } else {
                1.0 / cols_u32 as f32
            },
            entropy_epsilon: SPIRAL_ENTROPY_EPSILON,
            _pad: 0.0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_spiral.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.softmax_spiral_bind_group(
            softmax,
            mask,
            &spiral_buffer,
            &metrics_buffer,
            &params_buffer,
        )?;

        Some(SpiralConsensusResources {
            rows: rows_u32,
            bind_group,
            pipeline: Arc::clone(pipeline),
            spiral_buffer,
            metrics_buffer,
            _params_buffer: params_buffer,
            ramanujan_ratio,
            ramanujan_delta,
        })
    }
    fn bayesian_refine_softmax_score(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
    ) -> Option<SoftmaxBayesEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let mut weight = 0.0f64;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for (index, record) in history
            .iter()
            .rev()
            .filter(|entry| entry.variant == variant)
            .take(SOFTMAX_HISTORY_LIMIT)
            .enumerate()
        {
            let decay = (f64::from(GOLDEN_RATIO)).powf(-(index as f64) / 2.0);
            let z_bias = record
                .zmetrics
                .map(|metrics| {
                    let ratio_bias = 1.0 - (metrics.ramanujan_ratio as f64 - 1.0).abs().min(1.0);
                    1.0 + (metrics.spiral_flux as f64) * 0.5
                        + (metrics.focus as f64) * 0.25
                        + (metrics.leech_enrichment as f64) * 0.25
                        + ratio_bias * 0.2
                })
                .unwrap_or(1.0);
            let w = decay * z_bias;
            weight += w;
            sum += record.score_ms * w;
            sum_sq += record.score_ms * record.score_ms * w;
        }

        if weight <= f64::EPSILON {
            weight = f64::from(GOLDEN_RATIO);
            sum = raw_ms * weight;
            sum_sq = raw_ms * raw_ms * weight;
        }

        let prior_ms = (sum / weight).max(0.0);
        let mut prior_var = (sum_sq / weight) - prior_ms * prior_ms;
        prior_var = prior_var.max((prior_ms * 0.05).max(0.05).powi(2));

        let focus_boost = projection
            .map(|metrics| (metrics.focus as f64 + metrics.spiral_flux as f64).max(0.0))
            .unwrap_or(0.0);
        let leech_boost = projection
            .map(|metrics| metrics.leech_enrichment as f64)
            .unwrap_or(0.0);
        let ratio_harmonic = projection
            .map(|metrics| 1.0 - (metrics.ramanujan_ratio as f64 - 1.0).abs().min(1.0))
            .unwrap_or(0.0);
        let sample_weight =
            (f64::from(GOLDEN_RATIO) + focus_boost + leech_boost + ratio_harmonic).max(1.0);
        let measurement_var = (raw_ms * 0.08).max(0.05).powi(2);
        let posterior_precision = weight / prior_var + sample_weight / measurement_var;
        let posterior_var = if posterior_precision <= f64::EPSILON {
            prior_var
        } else {
            posterior_precision.recip()
        };
        let posterior_ms = (prior_ms * weight / prior_var
            + raw_ms * sample_weight / measurement_var)
            * posterior_var;
        let deviation = posterior_var.sqrt();
        let credible_low_ms = (posterior_ms - deviation).max(0.0);
        let credible_high_ms = posterior_ms + deviation;
        let combined_weight = weight + sample_weight;
        let confidence =
            (combined_weight / (combined_weight + f64::from(GOLDEN_RATIO))).clamp(0.0, 1.0) as f32;

        Some(SoftmaxBayesEvidence::new(
            posterior_ms,
            prior_ms,
            confidence,
            credible_low_ms,
            credible_high_ms,
        ))
    }
    fn metropolis_multi_try_softmax(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
    ) -> Option<SoftmaxMetropolisEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let candidate_focus = projection.map(|m| m.focus).unwrap_or(0.5);
        let candidate_flux = projection.map(|m| m.spiral_flux).unwrap_or(0.0);
        if history.is_empty() {
            return Some(SoftmaxMetropolisEvidence::identity(
                raw_ms,
                candidate_focus,
                candidate_flux,
            ));
        }

        let mut proposals = Vec::new();
        let norm_base = (raw_ms.abs() + SOFTMAX_METROPOLIS_TEMPERATURE).max(1.0);
        for record in history.into_iter().rev() {
            let metrics = match record.zmetrics {
                Some(metrics) => metrics,
                None => continue,
            };
            let affinity = zspace_affinity(projection, &metrics, variant == record.variant);
            let delta_ms = record.score_ms - raw_ms;
            let exponent = (-(delta_ms / norm_base)).clamp(-20.0, 20.0);
            let base = exponent.exp();
            let weight = base * (0.25 + 0.75 * affinity as f64);
            proposals.push((record.score_ms, metrics, weight, affinity as f64));
        }

        if proposals.is_empty() {
            return Some(SoftmaxMetropolisEvidence::identity(
                raw_ms,
                candidate_focus,
                candidate_flux,
            ));
        }

        proposals.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        proposals.truncate(SOFTMAX_METROPOLIS_TRIES);
        let tries = proposals.len() as u32;

        let mut total_weight = 1.0f64;
        let mut weighted_ms = raw_ms;
        let mut focus_sum = candidate_focus as f64;
        let mut flux_sum = candidate_flux as f64;
        let mut acceptance_sum = 0.0f64;
        for (score_ms, metrics, weight, affinity) in proposals.iter() {
            total_weight += *weight;
            weighted_ms += *weight * *score_ms;
            focus_sum += *weight * metrics.focus as f64;
            flux_sum += *weight * metrics.spiral_flux as f64;
            let delta_ms = raw_ms - *score_ms;
            let exponent = (delta_ms / norm_base).clamp(-20.0, 20.0);
            let logistic = 1.0 / (1.0 + (-exponent).exp());
            let component = logistic * (0.5 + 0.5 * affinity);
            acceptance_sum += component;
        }

        if tries == 0 {
            return Some(SoftmaxMetropolisEvidence::identity(
                raw_ms,
                candidate_focus,
                candidate_flux,
            ));
        }

        let acceptance = (acceptance_sum / tries as f64).clamp(0.0, 1.0) as f32;
        let expected_ms = (weighted_ms / total_weight).max(0.0);
        let proposal_focus = (focus_sum / total_weight).clamp(0.0, 1.0) as f32;
        let proposal_flux = (flux_sum / total_weight).clamp(0.0, 1.0) as f32;

        Some(SoftmaxMetropolisEvidence::new(
            acceptance,
            expected_ms,
            tries,
            proposal_focus,
            proposal_flux,
        ))
    }
    fn spiral_anneal_softmax(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
        bayes: Option<&SoftmaxBayesEvidence>,
        metropolis: Option<&SoftmaxMetropolisEvidence>,
    ) -> Option<SoftmaxSpiralAnnealEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let base_ms = bayes.map(|b| b.posterior_ms).unwrap_or(raw_ms);
        let focus = projection.map(|m| m.focus).unwrap_or(0.5);
        let flux = projection.map(|m| m.spiral_flux).unwrap_or(0.0);
        let swirl = projection.map(|m| m.swirl.abs()).unwrap_or(0.0);
        let leech = projection.map(|m| m.leech_enrichment).unwrap_or(0.0);
        let ratio_harmony = projection
            .map(|m| 1.0 - (m.ramanujan_ratio - 1.0).abs().min(1.0))
            .unwrap_or(0.5);
        let ratio_delta = projection.map(|m| m.ramanujan_delta).unwrap_or(0.0);
        let acceptance = metropolis.map(|m| m.acceptance).unwrap_or(0.0);
        let refreshes = metropolis.map(|m| m.tries).unwrap_or(0);

        let mut weighted_delta = 0.0f64;
        let mut weight_sum = 0.0f64;
        let mut dispersion = 0.0f64;
        let mut dispersion_weight = 0.0f64;

        for (index, record) in history
            .iter()
            .rev()
            .take(SOFTMAX_ANNEAL_HISTORY)
            .enumerate()
        {
            let decay = (f64::from(GOLDEN_RATIO)).powf(-(index as f64) / 1.5);
            let affinity = if record.variant == variant {
                1.0
            } else {
                record
                    .zmetrics
                    .map(|metrics| zspace_affinity(projection, &metrics, false) as f64)
                    .unwrap_or(0.35)
            };
            let weight = decay * (0.5 + 0.5 * affinity);
            weighted_delta += weight * (record.score_ms - base_ms);
            weight_sum += weight;

            if record.variant == variant {
                dispersion += weight * (record.score_ms - base_ms).abs();
                dispersion_weight += weight;
            }
        }

        if weight_sum <= f64::EPSILON {
            return Some(SoftmaxSpiralAnnealEvidence::identity(base_ms));
        }

        let drift = weighted_delta / weight_sum;
        let variant_dispersion = if dispersion_weight > f64::EPSILON {
            (dispersion / dispersion_weight).max(0.0)
        } else {
            0.0
        };

        let acceptance_factor =
            (0.6 + 0.4 * f64::from(acceptance) + 0.2 * f64::from(leech)).clamp(0.4, 1.2);
        let swirl_factor = (1.0 + f64::from(swirl).min(1.0) * 0.75).clamp(1.0, 1.75);
        let ratio_factor =
            (1.0 + ratio_harmony as f64 * 0.2 - f64::from(ratio_delta) * 0.1).clamp(0.8, 1.2);
        let focus_cool = (1.0 - f64::from(focus).clamp(0.0, 1.0)).clamp(0.0, 1.0);

        let base_temp = SOFTMAX_ANNEAL_MIN_TEMP
            + (SOFTMAX_ANNEAL_MAX_TEMP - SOFTMAX_ANNEAL_MIN_TEMP) * focus_cool;
        let temp = (base_temp * swirl_factor * acceptance_factor * ratio_factor)
            .clamp(SOFTMAX_ANNEAL_MIN_TEMP, SOFTMAX_ANNEAL_MAX_TEMP);

        let flux_correction =
            1.0 - (f64::from(flux).clamp(0.0, 1.0) * 0.3 + f64::from(leech).clamp(0.0, 1.0) * 0.2);
        let annealed_ms = (base_ms + drift * flux_correction).max(0.0);

        let exploration_mass = ((focus_cool
            + f64::from(flux).clamp(0.0, 1.0) * 0.5
            + f64::from(swirl).min(1.0) * 0.25
            + f64::from(leech).clamp(0.0, 1.0) * 0.2
            + ratio_harmony as f64 * 0.2)
            .clamp(0.0, 1.6))
        .min(1.0) as f32;

        let entropy = ((variant_dispersion / (base_ms + 1e-6)) * 0.6
            + f64::from(flux).clamp(0.0, 1.0) * 0.25
            + f64::from(leech).clamp(0.0, 1.0) * 0.1
            + f64::from(ratio_delta.min(1.0)) * 0.15)
            .clamp(0.0, 1.0) as f32;

        Some(SoftmaxSpiralAnnealEvidence::new(
            temp as f32,
            annealed_ms,
            exploration_mass,
            entropy,
            refreshes,
        ))
    }
    fn spiral_consensus_softmax(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
        bayes: Option<&SoftmaxBayesEvidence>,
        metropolis: Option<&SoftmaxMetropolisEvidence>,
        anneal: Option<&SoftmaxSpiralAnnealEvidence>,
    ) -> Option<SoftmaxSpiralConsensusEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let focus = projection.map(|m| m.focus).unwrap_or(0.5);
        let flux = projection.map(|m| m.spiral_flux).unwrap_or(0.0);
        let swirl = projection.map(|m| m.swirl).unwrap_or(0.0);
        let leech = projection.map(|m| m.leech_enrichment).unwrap_or(0.0);
        let ratio_alignment = projection
            .map(|m| 1.0 - (m.ramanujan_ratio - 1.0).abs().min(1.0))
            .unwrap_or(0.5);
        let ratio_delta = projection.map(|m| m.ramanujan_delta).unwrap_or(0.0);

        let mut harmony_sum = 0.0f64;
        let mut harmony_weight = 0.0f64;
        for (index, record) in history
            .iter()
            .rev()
            .take(SOFTMAX_CONSENSUS_HISTORY)
            .enumerate()
        {
            let decay = (f64::from(GOLDEN_RATIO)).powf(-(index as f64) / 3.0);
            let affinity = record
                .zmetrics
                .map(|metrics| {
                    zspace_affinity(projection, &metrics, record.variant == variant) as f64
                })
                .unwrap_or(0.5);
            harmony_sum += decay * affinity;
            harmony_weight += decay;
        }

        let harmony = if harmony_weight > f64::EPSILON {
            (harmony_sum / harmony_weight).clamp(0.0, 1.0) as f32
        } else {
            ((focus + flux + leech + ratio_alignment) * 0.25).clamp(0.0, 1.0)
        };

        let raw_weight =
            0.35 + 0.25 * f64::from(focus) + 0.2 * f64::from(leech) + 0.15 * ratio_alignment as f64;
        let mut weighted_ms = raw_ms.max(0.0) * raw_weight;
        let mut total_weight = raw_weight;
        let mut bayes_w = 0.0f64;
        let mut metro_w = 0.0f64;
        let mut anneal_w = 0.0f64;

        if let Some(bayes) = bayes {
            let w = 0.45 + 0.5 * f64::from(harmony) + 0.15 * ratio_alignment as f64;
            weighted_ms += bayes.posterior_ms.max(0.0) * w;
            total_weight += w;
            bayes_w = w;
        }

        if let Some(mtm) = metropolis {
            let w = 0.3 + 0.4 * f64::from(mtm.acceptance) + 0.2 * f64::from(leech);
            weighted_ms += mtm.expected_ms.max(0.0) * w;
            total_weight += w;
            metro_w = w;
        }

        if let Some(anneal) = anneal {
            let w = 0.25 + 0.3 * f64::from(anneal.exploration_mass) + 0.15 * ratio_alignment as f64;
            weighted_ms += anneal.annealed_ms.max(0.0) * w;
            total_weight += w;
            anneal_w = w;
        }

        if total_weight <= f64::EPSILON {
            return Some(SoftmaxSpiralConsensusEvidence::identity(
                raw_ms, focus, flux,
            ));
        }

        let consensus_ms = (weighted_ms / total_weight).max(0.0);
        let synergy = ((focus + flux + harmony + leech + ratio_alignment) / 5.0).clamp(0.0, 1.0);
        let z_bias = ((focus + flux + swirl.abs() + leech) * 0.25 + ratio_alignment * 0.2
            - ratio_delta.min(1.0) * 0.1)
            .clamp(0.0, 1.0);

        Some(SoftmaxSpiralConsensusEvidence::new(
            consensus_ms,
            synergy,
            z_bias,
            (bayes_w / total_weight) as f32,
            (metro_w / total_weight) as f32,
            (anneal_w / total_weight) as f32,
            harmony,
        ))
    }
    fn fused_gelu_back_bind_group(
        &self,
        z: &Buffer,
        g: &Buffer,
        gz_out: &Buffer,
        dr: &Buffer,
        partials: &Buffer,
        uniforms: &Buffer,
    ) -> BindGroup {
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_gelu_back.bind_group"),
            layout: &self.fused_gelu_back_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: g.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gz_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        })
    }

    fn reduce_db_bind_group(&self, partials: &Buffer, db: &Buffer, uniforms: &Buffer) -> BindGroup {
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.reduce_db.bind_group"),
            layout: &self.reduce_db_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: db.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        })
    }

    fn fused_conv_pipeline_for(&self, config: TileConfig) -> Result<Arc<ComputePipeline>, String> {
        let mut pipelines = self
            .fused_conv_pipelines
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(pipeline) = pipelines.get(&config) {
            return Ok(pipeline.clone());
        }

        let shader_source = instantiate_tile_template(FUSED_CONV_WGSL_TEMPLATE, config);
        let shader_label = format!(
            "st.tensor.wgpu_dense.fused_conv_shader.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let shader = create_wgsl_module(self.device(), &shader_label, shader_source.as_str())
            .map_err(|err| err.to_string())?;
        let pipeline_label = format!(
            "st.tensor.wgpu_dense.fused_conv_pipeline.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let pipeline = Arc::new(
            create_compute_pipeline(
                self.device(),
                &pipeline_label,
                Some(&self.fused_conv_pipeline_layout),
                &shader,
                "main",
            )
            .map_err(|err| err.to_string())?,
        );
        pipelines.insert(config, pipeline.clone());
        Ok(pipeline)
    }

    #[cfg(any())]
    fn softmax_spiral_bind_group(
        &self,
        softmax: &Buffer,
        mask: &Buffer,
        spiral: &Buffer,
        metrics: &Buffer,
        params: &Buffer,
    ) -> Option<BindGroup> {
        let layout = self.softmax_spiral_layout.as_ref()?;
        let descriptor = wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_spiral.bind_group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: softmax.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mask.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: spiral.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: metrics.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        };
        Some(self.device().create_bind_group(&descriptor))
    }

    #[cfg(any())]
    #[cfg(any())]
    fn prepare_spiral_consensus(
        &self,
        rows: usize,
        cols: usize,
        layout: &SoftmaxLayoutDesc,
        softmax: &Buffer,
        mask: &Buffer,
    ) -> Option<SpiralConsensusResources> {
        let pipeline = self.softmax_spiral_pipeline.as_ref()?;
        if rows == 0 || cols == 0 {
            return None;
        }

        let rows_u32 = u32::try_from(rows).ok()?;
        let cols_u32 = u32::try_from(cols).ok()?;
        let elements = rows.checked_mul(cols)?;
        let device = self.device();

        let spiral_buffer = allocate_output(
            device,
            "st.tensor.wgpu_dense.softmax_spiral.output",
            elements,
        );
        let metrics_len = rows.checked_mul(4)?;
        let metrics_size = (metrics_len * std::mem::size_of::<f32>()) as u64;
        let metrics_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_spiral.metrics"),
            size: metrics_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let approximation = ramanujan_pi(SPIRAL_PROJECTOR_RAMANUJAN_ITERS).max(f64::EPSILON);
        let pi = std::f64::consts::PI;
        let ramanujan_ratio = pi / approximation;
        let ramanujan_delta = (approximation - pi).abs();
        let sqrt_rank = (SPIRAL_PROJECTOR_RANK.max(1) as f64).sqrt();
        let leech_scale =
            SPIRAL_PROJECTOR_WEIGHT * SPIRAL_LEECH_PACKING_DENSITY * sqrt_rank * ramanujan_ratio;

        let params = SpiralConsensusParams {
            rows: rows_u32,
            cols: cols_u32,
            soft_stride: layout.out_stride,
            mask_stride: layout.out_stride,
            spiral_stride: layout.out_stride,
            chimera_tile: layout.chimera_tile,
            chimera_stripes: layout.chimera_stripes,
            flags: layout.flags,
            phi: GOLDEN_RATIO,
            phi_conjugate: GOLDEN_RATIO_CONJUGATE,
            phi_bias: GOLDEN_RATIO_BIAS,
            leech_scale: leech_scale as f32,
            ramanujan_ratio: ramanujan_ratio as f32,
            inv_cols: if cols_u32 == 0 {
                0.0
            } else {
                1.0 / cols_u32 as f32
            },
            entropy_epsilon: SPIRAL_ENTROPY_EPSILON,
            _pad: 0.0,
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_spiral.params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.softmax_spiral_bind_group(
            softmax,
            mask,
            &spiral_buffer,
            &metrics_buffer,
            &params_buffer,
        )?;

        Some(SpiralConsensusResources {
            rows: rows_u32,
            bind_group,
            pipeline: Arc::clone(pipeline),
            spiral_buffer,
            metrics_buffer,
            _params_buffer: params_buffer,
            ramanujan_ratio,
            ramanujan_delta,
        })
    }

    #[cfg(any())]
    fn bayesian_refine_softmax_score(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
    ) -> Option<SoftmaxBayesEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let mut weight = 0.0f64;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for (index, record) in history
            .iter()
            .rev()
            .filter(|entry| entry.variant == variant)
            .take(SOFTMAX_HISTORY_LIMIT)
            .enumerate()
        {
            let decay = (f64::from(GOLDEN_RATIO)).powf(-(index as f64) / 2.0);
            let z_bias = record
                .zmetrics
                .map(|metrics| {
                    let ratio_bias = 1.0 - (metrics.ramanujan_ratio as f64 - 1.0).abs().min(1.0);
                    1.0 + (metrics.spiral_flux as f64) * 0.5
                        + (metrics.focus as f64) * 0.25
                        + (metrics.leech_enrichment as f64) * 0.25
                        + ratio_bias * 0.2
                })
                .unwrap_or(1.0);
            let w = decay * z_bias;
            weight += w;
            sum += record.score_ms * w;
            sum_sq += record.score_ms * record.score_ms * w;
        }

        if weight <= f64::EPSILON {
            weight = f64::from(GOLDEN_RATIO);
            sum = raw_ms * weight;
            sum_sq = raw_ms * raw_ms * weight;
        }

        let prior_ms = (sum / weight).max(0.0);
        let mut prior_var = (sum_sq / weight) - prior_ms * prior_ms;
        prior_var = prior_var.max((prior_ms * 0.05).max(0.05).powi(2));

        let focus_boost = projection
            .map(|metrics| (metrics.focus as f64 + metrics.spiral_flux as f64).max(0.0))
            .unwrap_or(0.0);
        let leech_boost = projection
            .map(|metrics| metrics.leech_enrichment as f64)
            .unwrap_or(0.0);
        let ratio_harmonic = projection
            .map(|metrics| 1.0 - (metrics.ramanujan_ratio as f64 - 1.0).abs().min(1.0))
            .unwrap_or(0.0);
        let sample_weight =
            (f64::from(GOLDEN_RATIO) + focus_boost + leech_boost + ratio_harmonic).max(1.0);
        let measurement_var = (raw_ms * 0.08).max(0.05).powi(2);
        let posterior_precision = weight / prior_var + sample_weight / measurement_var;
        let posterior_var = if posterior_precision <= f64::EPSILON {
            prior_var
        } else {
            posterior_precision.recip()
        };
        let posterior_ms = (prior_ms * weight / prior_var
            + raw_ms * sample_weight / measurement_var)
            * posterior_var;
        let deviation = posterior_var.sqrt();
        let credible_low_ms = (posterior_ms - deviation).max(0.0);
        let credible_high_ms = posterior_ms + deviation;
        let combined_weight = weight + sample_weight;
        let confidence =
            (combined_weight / (combined_weight + f64::from(GOLDEN_RATIO))).clamp(0.0, 1.0) as f32;

        Some(SoftmaxBayesEvidence::new(
            posterior_ms,
            prior_ms,
            confidence,
            credible_low_ms,
            credible_high_ms,
        ))
    }

    #[cfg(any())]
    fn metropolis_multi_try_softmax(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
    ) -> Option<SoftmaxMetropolisEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let candidate_focus = projection.map(|m| m.focus).unwrap_or(0.5);
        let candidate_flux = projection.map(|m| m.spiral_flux).unwrap_or(0.0);
        if history.is_empty() {
            return Some(SoftmaxMetropolisEvidence::identity(
                raw_ms,
                candidate_focus,
                candidate_flux,
            ));
        }

        let mut proposals = Vec::new();
        let norm_base = (raw_ms.abs() + SOFTMAX_METROPOLIS_TEMPERATURE).max(1.0);
        for record in history.into_iter().rev() {
            let metrics = match record.zmetrics {
                Some(metrics) => metrics,
                None => continue,
            };
            let affinity = zspace_affinity(projection, &metrics, variant == record.variant);
            let delta_ms = record.score_ms - raw_ms;
            let exponent = (-(delta_ms / norm_base)).clamp(-20.0, 20.0);
            let base = exponent.exp();
            let weight = base * (0.25 + 0.75 * affinity as f64);
            proposals.push((record.score_ms, metrics, weight, affinity as f64));
        }

        if proposals.is_empty() {
            return Some(SoftmaxMetropolisEvidence::identity(
                raw_ms,
                candidate_focus,
                candidate_flux,
            ));
        }

        proposals.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        proposals.truncate(SOFTMAX_METROPOLIS_TRIES);
        let tries = proposals.len() as u32;

        let mut total_weight = 1.0f64;
        let mut weighted_ms = raw_ms;
        let mut focus_sum = candidate_focus as f64;
        let mut flux_sum = candidate_flux as f64;
        let mut acceptance_sum = 0.0f64;
        for (score_ms, metrics, weight, affinity) in proposals.iter() {
            total_weight += *weight;
            weighted_ms += *weight * *score_ms;
            focus_sum += *weight * metrics.focus as f64;
            flux_sum += *weight * metrics.spiral_flux as f64;
            let delta_ms = raw_ms - *score_ms;
            let exponent = (delta_ms / norm_base).clamp(-20.0, 20.0);
            let logistic = 1.0 / (1.0 + (-exponent).exp());
            let component = logistic * (0.5 + 0.5 * affinity);
            acceptance_sum += component;
        }

        if tries == 0 {
            return Some(SoftmaxMetropolisEvidence::identity(
                raw_ms,
                candidate_focus,
                candidate_flux,
            ));
        }

        let acceptance = (acceptance_sum / tries as f64).clamp(0.0, 1.0) as f32;
        let expected_ms = (weighted_ms / total_weight).max(0.0);
        let proposal_focus = (focus_sum / total_weight).clamp(0.0, 1.0) as f32;
        let proposal_flux = (flux_sum / total_weight).clamp(0.0, 1.0) as f32;

        Some(SoftmaxMetropolisEvidence::new(
            acceptance,
            expected_ms,
            tries,
            proposal_focus,
            proposal_flux,
        ))
    }

    #[cfg(any())]
    fn spiral_anneal_softmax(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
        bayes: Option<&SoftmaxBayesEvidence>,
        metropolis: Option<&SoftmaxMetropolisEvidence>,
    ) -> Option<SoftmaxSpiralAnnealEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let base_ms = bayes.map(|b| b.posterior_ms).unwrap_or(raw_ms);
        let focus = projection.map(|m| m.focus).unwrap_or(0.5);
        let flux = projection.map(|m| m.spiral_flux).unwrap_or(0.0);
        let swirl = projection.map(|m| m.swirl.abs()).unwrap_or(0.0);
        let leech = projection.map(|m| m.leech_enrichment).unwrap_or(0.0);
        let ratio_harmony = projection
            .map(|m| 1.0 - (m.ramanujan_ratio - 1.0).abs().min(1.0))
            .unwrap_or(0.5);
        let ratio_delta = projection.map(|m| m.ramanujan_delta).unwrap_or(0.0);
        let acceptance = metropolis.map(|m| m.acceptance).unwrap_or(0.0);
        let refreshes = metropolis.map(|m| m.tries).unwrap_or(0);

        let mut weighted_delta = 0.0f64;
        let mut weight_sum = 0.0f64;
        let mut dispersion = 0.0f64;
        let mut dispersion_weight = 0.0f64;

        for (index, record) in history
            .iter()
            .rev()
            .take(SOFTMAX_ANNEAL_HISTORY)
            .enumerate()
        {
            let decay = (f64::from(GOLDEN_RATIO)).powf(-(index as f64) / 1.5);
            let affinity = if record.variant == variant {
                1.0
            } else {
                record
                    .zmetrics
                    .map(|metrics| zspace_affinity(projection, &metrics, false) as f64)
                    .unwrap_or(0.35)
            };
            let weight = decay * (0.5 + 0.5 * affinity);
            weighted_delta += weight * (record.score_ms - base_ms);
            weight_sum += weight;

            if record.variant == variant {
                dispersion += weight * (record.score_ms - base_ms).abs();
                dispersion_weight += weight;
            }
        }

        if weight_sum <= f64::EPSILON {
            return Some(SoftmaxSpiralAnnealEvidence::identity(base_ms));
        }

        let drift = weighted_delta / weight_sum;
        let variant_dispersion = if dispersion_weight > f64::EPSILON {
            (dispersion / dispersion_weight).max(0.0)
        } else {
            0.0
        };

        let acceptance_factor =
            (0.6 + 0.4 * f64::from(acceptance) + 0.2 * f64::from(leech)).clamp(0.4, 1.2);
        let swirl_factor = (1.0 + f64::from(swirl).min(1.0) * 0.75).clamp(1.0, 1.75);
        let ratio_factor =
            (1.0 + ratio_harmony as f64 * 0.2 - f64::from(ratio_delta) * 0.1).clamp(0.8, 1.2);
        let focus_cool = (1.0 - f64::from(focus).clamp(0.0, 1.0)).clamp(0.0, 1.0);

        let base_temp = SOFTMAX_ANNEAL_MIN_TEMP
            + (SOFTMAX_ANNEAL_MAX_TEMP - SOFTMAX_ANNEAL_MIN_TEMP) * focus_cool;
        let temp = (base_temp * swirl_factor * acceptance_factor * ratio_factor)
            .clamp(SOFTMAX_ANNEAL_MIN_TEMP, SOFTMAX_ANNEAL_MAX_TEMP);

        let flux_correction =
            1.0 - (f64::from(flux).clamp(0.0, 1.0) * 0.3 + f64::from(leech).clamp(0.0, 1.0) * 0.2);
        let annealed_ms = (base_ms + drift * flux_correction).max(0.0);

        let exploration_mass = ((focus_cool
            + f64::from(flux).clamp(0.0, 1.0) * 0.5
            + f64::from(swirl).min(1.0) * 0.25
            + f64::from(leech).clamp(0.0, 1.0) * 0.2
            + ratio_harmony as f64 * 0.2)
            .clamp(0.0, 1.6))
        .min(1.0) as f32;

        let entropy = ((variant_dispersion / (base_ms + 1e-6)) * 0.6
            + f64::from(flux).clamp(0.0, 1.0) * 0.25
            + f64::from(leech).clamp(0.0, 1.0) * 0.1
            + f64::from(ratio_delta.min(1.0)) * 0.15)
            .clamp(0.0, 1.0) as f32;

        Some(SoftmaxSpiralAnnealEvidence::new(
            temp as f32,
            annealed_ms,
            exploration_mass,
            entropy,
            refreshes,
        ))
    }

    #[cfg(any())]
    fn spiral_consensus_softmax(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
        bayes: Option<&SoftmaxBayesEvidence>,
        metropolis: Option<&SoftmaxMetropolisEvidence>,
        anneal: Option<&SoftmaxSpiralAnnealEvidence>,
    ) -> Option<SoftmaxSpiralConsensusEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let focus = projection.map(|m| m.focus).unwrap_or(0.5);
        let flux = projection.map(|m| m.spiral_flux).unwrap_or(0.0);
        let swirl = projection.map(|m| m.swirl).unwrap_or(0.0);
        let leech = projection.map(|m| m.leech_enrichment).unwrap_or(0.0);
        let ratio_alignment = projection
            .map(|m| 1.0 - (m.ramanujan_ratio - 1.0).abs().min(1.0))
            .unwrap_or(0.5);
        let ratio_delta = projection.map(|m| m.ramanujan_delta).unwrap_or(0.0);

        let mut harmony_sum = 0.0f64;
        let mut harmony_weight = 0.0f64;
        for (index, record) in history
            .iter()
            .rev()
            .take(SOFTMAX_CONSENSUS_HISTORY)
            .enumerate()
        {
            let decay = (f64::from(GOLDEN_RATIO)).powf(-(index as f64) / 3.0);
            let affinity = record
                .zmetrics
                .map(|metrics| {
                    zspace_affinity(projection, &metrics, record.variant == variant) as f64
                })
                .unwrap_or(0.5);
            harmony_sum += decay * affinity;
            harmony_weight += decay;
        }

        let harmony = if harmony_weight > f64::EPSILON {
            (harmony_sum / harmony_weight).clamp(0.0, 1.0) as f32
        } else {
            ((focus + flux + leech + ratio_alignment) * 0.25).clamp(0.0, 1.0)
        };

        let raw_weight =
            0.35 + 0.25 * f64::from(focus) + 0.2 * f64::from(leech) + 0.15 * ratio_alignment as f64;
        let mut weighted_ms = raw_ms.max(0.0) * raw_weight;
        let mut total_weight = raw_weight;
        let mut bayes_w = 0.0f64;
        let mut metro_w = 0.0f64;
        let mut anneal_w = 0.0f64;

        if let Some(bayes) = bayes {
            let w = 0.45 + 0.5 * f64::from(harmony) + 0.15 * ratio_alignment as f64;
            weighted_ms += bayes.posterior_ms.max(0.0) * w;
            total_weight += w;
            bayes_w = w;
        }

        if let Some(mtm) = metropolis {
            let w = 0.3 + 0.4 * f64::from(mtm.acceptance) + 0.2 * f64::from(leech);
            weighted_ms += mtm.expected_ms.max(0.0) * w;
            total_weight += w;
            metro_w = w;
        }

        if let Some(anneal) = anneal {
            let w = 0.25 + 0.3 * f64::from(anneal.exploration_mass) + 0.15 * ratio_alignment as f64;
            weighted_ms += anneal.annealed_ms.max(0.0) * w;
            total_weight += w;
            anneal_w = w;
        }

        if total_weight <= f64::EPSILON {
            return Some(SoftmaxSpiralConsensusEvidence::identity(
                raw_ms, focus, flux,
            ));
        }

        let consensus_ms = (weighted_ms / total_weight).max(0.0);
        let synergy = ((focus + flux + harmony + leech + ratio_alignment) / 5.0).clamp(0.0, 1.0);
        let z_bias = ((focus + flux + swirl.abs() + leech) * 0.25 + ratio_alignment * 0.2
            - ratio_delta.min(1.0) * 0.1)
            .clamp(0.0, 1.0);

        Some(SoftmaxSpiralConsensusEvidence::new(
            consensus_ms,
            synergy,
            z_bias,
            (bayes_w / total_weight) as f32,
            (metro_w / total_weight) as f32,
            (anneal_w / total_weight) as f32,
            harmony,
        ))
    }

    #[cfg(any())]
    fn bayesian_refine_softmax_score(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
    ) -> Option<SoftmaxBayesEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let mut weight = 0.0f64;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for (index, record) in history
            .iter()
            .rev()
            .filter(|entry| entry.variant == variant)
            .take(SOFTMAX_HISTORY_LIMIT)
            .enumerate()
        {
            let decay = (f64::from(GOLDEN_RATIO)).powf(-(index as f64) / 2.0);
            let z_bias = record
                .zmetrics
                .map(|metrics| {
                    let ratio_bias = 1.0 - (metrics.ramanujan_ratio as f64 - 1.0).abs().min(1.0);
                    1.0 + (metrics.spiral_flux as f64) * 0.5
                        + (metrics.focus as f64) * 0.25
                        + (metrics.leech_enrichment as f64) * 0.25
                        + ratio_bias * 0.2
                })
                .unwrap_or(1.0);
            let w = decay * z_bias;
            weight += w;
            sum += record.score_ms * w;
            sum_sq += record.score_ms * record.score_ms * w;
        }

        if weight <= f64::EPSILON {
            weight = f64::from(GOLDEN_RATIO);
            sum = raw_ms * weight;
            sum_sq = raw_ms * raw_ms * weight;
        }

        let prior_ms = (sum / weight).max(0.0);
        let mut prior_var = (sum_sq / weight) - prior_ms * prior_ms;
        prior_var = prior_var.max((prior_ms * 0.05).max(0.05).powi(2));

        let focus_boost = projection
            .map(|metrics| (metrics.focus as f64 + metrics.spiral_flux as f64).max(0.0))
            .unwrap_or(0.0);
        let leech_boost = projection
            .map(|metrics| metrics.leech_enrichment as f64)
            .unwrap_or(0.0);
        let ratio_harmonic = projection
            .map(|metrics| 1.0 - (metrics.ramanujan_ratio as f64 - 1.0).abs().min(1.0))
            .unwrap_or(0.0);
        let sample_weight =
            (f64::from(GOLDEN_RATIO) + focus_boost + leech_boost + ratio_harmonic).max(1.0);
        let measurement_var = (raw_ms * 0.08).max(0.05).powi(2);
        let posterior_precision = weight / prior_var + sample_weight / measurement_var;
        let posterior_var = if posterior_precision <= f64::EPSILON {
            prior_var
        } else {
            posterior_precision.recip()
        };
        let posterior_ms = (prior_ms * weight / prior_var
            + raw_ms * sample_weight / measurement_var)
            * posterior_var;
        let deviation = posterior_var.sqrt();
        let credible_low_ms = (posterior_ms - deviation).max(0.0);
        let credible_high_ms = posterior_ms + deviation;
        let combined_weight = weight + sample_weight;
        let confidence =
            (combined_weight / (combined_weight + f64::from(GOLDEN_RATIO))).clamp(0.0, 1.0) as f32;

        Some(SoftmaxBayesEvidence::new(
            posterior_ms,
            prior_ms,
            confidence,
            credible_low_ms,
            credible_high_ms,
        ))
    }

    #[cfg(any())]
    fn metropolis_multi_try_softmax(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
    ) -> Option<SoftmaxMetropolisEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let candidate_focus = projection.map(|m| m.focus).unwrap_or(0.5);
        let candidate_flux = projection.map(|m| m.spiral_flux).unwrap_or(0.0);
        if history.is_empty() {
            return Some(SoftmaxMetropolisEvidence::identity(
                raw_ms,
                candidate_focus,
                candidate_flux,
            ));
        }

        let mut proposals = Vec::new();
        let norm_base = (raw_ms.abs() + SOFTMAX_METROPOLIS_TEMPERATURE).max(1.0);
        for record in history.into_iter().rev() {
            let metrics = match record.zmetrics {
                Some(metrics) => metrics,
                None => continue,
            };
            let affinity = zspace_affinity(projection, &metrics, variant == record.variant);
            let delta_ms = record.score_ms - raw_ms;
            let exponent = (-(delta_ms / norm_base)).clamp(-20.0, 20.0);
            let base = exponent.exp();
            let weight = base * (0.25 + 0.75 * affinity as f64);
            proposals.push((record.score_ms, metrics, weight, affinity as f64));
        }

        if proposals.is_empty() {
            return Some(SoftmaxMetropolisEvidence::identity(
                raw_ms,
                candidate_focus,
                candidate_flux,
            ));
        }

        proposals.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        proposals.truncate(SOFTMAX_METROPOLIS_TRIES);
        let tries = proposals.len() as u32;

        let mut total_weight = 1.0f64;
        let mut weighted_ms = raw_ms;
        let mut focus_sum = candidate_focus as f64;
        let mut flux_sum = candidate_flux as f64;
        let mut acceptance_sum = 0.0f64;
        for (score_ms, metrics, weight, affinity) in proposals.iter() {
            total_weight += *weight;
            weighted_ms += *weight * *score_ms;
            focus_sum += *weight * metrics.focus as f64;
            flux_sum += *weight * metrics.spiral_flux as f64;
            let delta_ms = raw_ms - *score_ms;
            let exponent = (delta_ms / norm_base).clamp(-20.0, 20.0);
            let logistic = 1.0 / (1.0 + (-exponent).exp());
            let component = logistic * (0.5 + 0.5 * affinity);
            acceptance_sum += component;
        }

        if tries == 0 {
            return Some(SoftmaxMetropolisEvidence::identity(
                raw_ms,
                candidate_focus,
                candidate_flux,
            ));
        }

        let acceptance = (acceptance_sum / tries as f64).clamp(0.0, 1.0) as f32;
        let expected_ms = (weighted_ms / total_weight).max(0.0);
        let proposal_focus = (focus_sum / total_weight).clamp(0.0, 1.0) as f32;
        let proposal_flux = (flux_sum / total_weight).clamp(0.0, 1.0) as f32;

        Some(SoftmaxMetropolisEvidence::new(
            acceptance,
            expected_ms,
            tries,
            proposal_focus,
            proposal_flux,
        ))
    }

    #[cfg(any())]
    fn spiral_anneal_softmax(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
        bayes: Option<&SoftmaxBayesEvidence>,
        metropolis: Option<&SoftmaxMetropolisEvidence>,
    ) -> Option<SoftmaxSpiralAnnealEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let base_ms = bayes.map(|b| b.posterior_ms).unwrap_or(raw_ms);
        let focus = projection.map(|m| m.focus).unwrap_or(0.5);
        let flux = projection.map(|m| m.spiral_flux).unwrap_or(0.0);
        let swirl = projection.map(|m| m.swirl.abs()).unwrap_or(0.0);
        let leech = projection.map(|m| m.leech_enrichment).unwrap_or(0.0);
        let ratio_harmony = projection
            .map(|m| 1.0 - (m.ramanujan_ratio - 1.0).abs().min(1.0))
            .unwrap_or(0.5);
        let ratio_delta = projection.map(|m| m.ramanujan_delta).unwrap_or(0.0);
        let acceptance = metropolis.map(|m| m.acceptance).unwrap_or(0.0);
        let refreshes = metropolis.map(|m| m.tries).unwrap_or(0);

        let mut weighted_delta = 0.0f64;
        let mut weight_sum = 0.0f64;
        let mut dispersion = 0.0f64;
        let mut dispersion_weight = 0.0f64;

        for (index, record) in history
            .iter()
            .rev()
            .take(SOFTMAX_ANNEAL_HISTORY)
            .enumerate()
        {
            let decay = (f64::from(GOLDEN_RATIO)).powf(-(index as f64) / 1.5);
            let affinity = if record.variant == variant {
                1.0
            } else {
                record
                    .zmetrics
                    .map(|metrics| zspace_affinity(projection, &metrics, false) as f64)
                    .unwrap_or(0.35)
            };
            let weight = decay * (0.5 + 0.5 * affinity);
            weighted_delta += weight * (record.score_ms - base_ms);
            weight_sum += weight;

            if record.variant == variant {
                dispersion += weight * (record.score_ms - base_ms).abs();
                dispersion_weight += weight;
            }
        }

        if weight_sum <= f64::EPSILON {
            return Some(SoftmaxSpiralAnnealEvidence::identity(base_ms));
        }

        let drift = weighted_delta / weight_sum;
        let variant_dispersion = if dispersion_weight > f64::EPSILON {
            (dispersion / dispersion_weight).max(0.0)
        } else {
            0.0
        };

        let acceptance_factor =
            (0.6 + 0.4 * f64::from(acceptance) + 0.2 * f64::from(leech)).clamp(0.4, 1.2);
        let swirl_factor = (1.0 + f64::from(swirl).min(1.0) * 0.75).clamp(1.0, 1.75);
        let ratio_factor =
            (1.0 + ratio_harmony as f64 * 0.2 - f64::from(ratio_delta) * 0.1).clamp(0.8, 1.2);
        let focus_cool = (1.0 - f64::from(focus).clamp(0.0, 1.0)).clamp(0.0, 1.0);

        let base_temp = SOFTMAX_ANNEAL_MIN_TEMP
            + (SOFTMAX_ANNEAL_MAX_TEMP - SOFTMAX_ANNEAL_MIN_TEMP) * focus_cool;
        let temp = (base_temp * swirl_factor * acceptance_factor * ratio_factor)
            .clamp(SOFTMAX_ANNEAL_MIN_TEMP, SOFTMAX_ANNEAL_MAX_TEMP);

        let flux_correction =
            1.0 - (f64::from(flux).clamp(0.0, 1.0) * 0.3 + f64::from(leech).clamp(0.0, 1.0) * 0.2);
        let annealed_ms = (base_ms + drift * flux_correction).max(0.0);

        let exploration_mass = ((focus_cool
            + f64::from(flux).clamp(0.0, 1.0) * 0.5
            + f64::from(swirl).min(1.0) * 0.25
            + f64::from(leech).clamp(0.0, 1.0) * 0.2
            + ratio_harmony as f64 * 0.2)
            .clamp(0.0, 1.6))
        .min(1.0) as f32;

        let entropy = ((variant_dispersion / (base_ms + 1e-6)) * 0.6
            + f64::from(flux).clamp(0.0, 1.0) * 0.25
            + f64::from(leech).clamp(0.0, 1.0) * 0.1
            + f64::from(ratio_delta.min(1.0)) * 0.15)
            .clamp(0.0, 1.0) as f32;

        Some(SoftmaxSpiralAnnealEvidence::new(
            temp as f32,
            annealed_ms,
            exploration_mass,
            entropy,
            refreshes,
        ))
    }

    #[cfg(any())]
    fn spiral_consensus_softmax(
        &self,
        variant: SoftmaxVariant,
        raw_ms: f64,
        projection: Option<&SoftmaxZProjectMetrics>,
        bayes: Option<&SoftmaxBayesEvidence>,
        metropolis: Option<&SoftmaxMetropolisEvidence>,
        anneal: Option<&SoftmaxSpiralAnnealEvidence>,
    ) -> Option<SoftmaxSpiralConsensusEvidence> {
        let history = self.softmax_history.lock().ok()?.clone();
        let focus = projection.map(|m| m.focus).unwrap_or(0.5);
        let flux = projection.map(|m| m.spiral_flux).unwrap_or(0.0);
        let swirl = projection.map(|m| m.swirl).unwrap_or(0.0);
        let leech = projection.map(|m| m.leech_enrichment).unwrap_or(0.0);
        let ratio_alignment = projection
            .map(|m| 1.0 - (m.ramanujan_ratio - 1.0).abs().min(1.0))
            .unwrap_or(0.5);
        let ratio_delta = projection.map(|m| m.ramanujan_delta).unwrap_or(0.0);

        let mut harmony_sum = 0.0f64;
        let mut harmony_weight = 0.0f64;
        for (index, record) in history
            .iter()
            .rev()
            .take(SOFTMAX_CONSENSUS_HISTORY)
            .enumerate()
        {
            let decay = (f64::from(GOLDEN_RATIO)).powf(-(index as f64) / 3.0);
            let affinity = record
                .zmetrics
                .map(|metrics| {
                    zspace_affinity(projection, &metrics, record.variant == variant) as f64
                })
                .unwrap_or(0.5);
            harmony_sum += decay * affinity;
            harmony_weight += decay;
        }

        let harmony = if harmony_weight > f64::EPSILON {
            (harmony_sum / harmony_weight).clamp(0.0, 1.0) as f32
        } else {
            ((focus + flux + leech + ratio_alignment) * 0.25).clamp(0.0, 1.0)
        };

        let raw_weight =
            0.35 + 0.25 * f64::from(focus) + 0.2 * f64::from(leech) + 0.15 * ratio_alignment as f64;
        let mut weighted_ms = raw_ms.max(0.0) * raw_weight;
        let mut total_weight = raw_weight;
        let mut bayes_w = 0.0f64;
        let mut metro_w = 0.0f64;
        let mut anneal_w = 0.0f64;

        if let Some(bayes) = bayes {
            let w = 0.45 + 0.5 * f64::from(harmony) + 0.15 * ratio_alignment as f64;
            weighted_ms += bayes.posterior_ms.max(0.0) * w;
            total_weight += w;
            bayes_w = w;
        }

        if let Some(mtm) = metropolis {
            let w = 0.3 + 0.4 * f64::from(mtm.acceptance) + 0.2 * f64::from(leech);
            weighted_ms += mtm.expected_ms.max(0.0) * w;
            total_weight += w;
            metro_w = w;
        }

        if let Some(anneal) = anneal {
            let w = 0.25 + 0.3 * f64::from(anneal.exploration_mass) + 0.15 * ratio_alignment as f64;
            weighted_ms += anneal.annealed_ms.max(0.0) * w;
            total_weight += w;
            anneal_w = w;
        }

        if total_weight <= f64::EPSILON {
            return Some(SoftmaxSpiralConsensusEvidence::identity(
                raw_ms, focus, flux,
            ));
        }

        let consensus_ms = (weighted_ms / total_weight).max(0.0);
        let synergy = ((focus + flux + harmony + leech + ratio_alignment) / 5.0).clamp(0.0, 1.0);
        let z_bias = ((focus + flux + swirl.abs() + leech) * 0.25 + ratio_alignment * 0.2
            - ratio_delta.min(1.0) * 0.1)
            .clamp(0.0, 1.0);

        Some(SoftmaxSpiralConsensusEvidence::new(
            consensus_ms,
            synergy,
            z_bias,
            (bayes_w / total_weight) as f32,
            (metro_w / total_weight) as f32,
            (anneal_w / total_weight) as f32,
            harmony,
        ))
    }

    #[cfg(any())]
    fn fused_gelu_back_bind_group(
        &self,
        z: &Buffer,
        g: &Buffer,
        gz_out: &Buffer,
        dr: &Buffer,
        partials: &Buffer,
        uniforms: &Buffer,
    ) -> BindGroup {
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_gelu_back.bind_group"),
            layout: &self.fused_gelu_back_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: z.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: g.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gz_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dr.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        })
    }

    #[cfg(any())]
    fn reduce_db_bind_group(&self, partials: &Buffer, db: &Buffer, uniforms: &Buffer) -> BindGroup {
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.reduce_db.bind_group"),
            layout: &self.reduce_db_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: db.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniforms.as_entire_binding(),
                },
            ],
        })
    }

    #[cfg(any())]
    fn fused_conv_pipeline_for(&self, config: TileConfig) -> Result<Arc<ComputePipeline>, String> {
        let mut pipelines = self
            .fused_conv_pipelines
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(pipeline) = pipelines.get(&config) {
            return Ok(pipeline.clone());
        }

        let shader_source = instantiate_tile_template(FUSED_CONV_WGSL_TEMPLATE, config);
        let shader_label = format!(
            "st.tensor.wgpu_dense.fused_conv_shader.tile{}x{}x{}",
            config.tile_m(),
            config.tile_n(),
            config.tile_k(),
        );
        let shader = create_wgsl_module(self.device(), &shader_label, shader_source.as_str())
            .map_err(|err| err.to_string())?;
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
                compilation_options: Default::default(),
            },
        ));
        pipelines.insert(config, pipeline.clone());
        Ok(pipeline)
    }

    fn fused_conv_bind_group(
        &self,
        input: &Buffer,
        weights: &Buffer,
        output: &Buffer,
        bias: &Buffer,
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
                    resource: bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    fn fused_grad_input_pipeline(&self) -> Result<Arc<ComputePipeline>, String> {
        if let Some(pipeline) = self.fused_grad_input_pipeline.get() {
            return Ok(pipeline.clone());
        }
        let source = FUSED_GRAD_INPUT_WGSL_TEMPLATE
            .replace("{tile_x}", &GRAD_INPUT_TILE_X.to_string())
            .replace("{tile_y}", &GRAD_INPUT_TILE_Y.to_string())
            .replace("{tile_z}", &GRAD_INPUT_TILE_Z.to_string())
            .replace(
                "{ramanujan_pi_6}",
                &format!("{:.*}", 18, ramanujan_pi(RAMANUJAN_PI_ITERATIONS) as f32),
            );
        let shader = create_wgsl_module(
            self.device(),
            "st.tensor.wgpu_dense.fused_grad_input",
            &source,
        )
        .map_err(|err| err.to_string())?;
        let pipeline = Arc::new(
            create_compute_pipeline(
                self.device(),
                "st.tensor.wgpu_dense.fused_grad_input.pipeline",
                Some(&self.fused_grad_input_pipeline_layout),
                &shader,
                "main",
            )
            .map_err(|err| err.to_string())?,
        );
        let _ = self.fused_grad_input_pipeline.set(pipeline.clone());
        Ok(pipeline)
    }

    fn fused_grad_input_bind_group(
        &self,
        grad_matrix: &Buffer,
        weights: &Buffer,
        output: &Buffer,
        params: &Buffer,
    ) -> BindGroup {
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.fused_grad_input.bind_group"),
            layout: &self.fused_grad_input_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grad_matrix.as_entire_binding(),
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

    fn ramanujan_pi_pipeline(&self) -> Result<Arc<ComputePipeline>, String> {
        if let Some(pipeline) = self.ramanujan_pipeline.get() {
            return Ok(pipeline.clone());
        }
        let shader = create_wgsl_module(
            self.device(),
            "st.tensor.wgpu_dense.ramanujan_pi",
            RAMANUJAN_PI_WGSL,
        )
        .map_err(|err| err.to_string())?;
        let pipeline = Arc::new(
            create_compute_pipeline(
                self.device(),
                "st.tensor.wgpu_dense.ramanujan_pi.pipeline",
                Some(&self.ramanujan_pipeline_layout),
                &shader,
                "main",
            )
            .map_err(|err| err.to_string())?,
        );
        let _ = self.ramanujan_pipeline.set(pipeline.clone());
        Ok(pipeline)
    }

    fn ramanujan_pi_bind_group(&self, output: &Buffer, params: &Buffer) -> BindGroup {
        self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("st.tensor.wgpu_dense.ramanujan_pi.bind_group"),
            layout: &self.ramanujan_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
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

#[cfg(all(test, feature = "wgpu_dense"))]
mod tests {
    use super::*;
    use crate::{Tensor, Tile};
    use naga::front::wgsl::parse_str;

    fn assert_parses(label: &str, source: &str) {
        parse_str(source).unwrap_or_else(|err| panic!("{label} failed: {err}"));
    }

    #[test]
    fn matmul_shader_wgsl_is_valid() {
        let key = PipelineKey::new(
            ScalarType::F32,
            TileConfig::new(16, 16, 16),
            false,
            false,
            false,
            0,
        );
        let source = generate_matmul_shader(&key);
        assert_parses("matmul f32", &source);

        let key_f16 = PipelineKey::new(
            ScalarType::F32,
            TileConfig::new(8, 8, 8),
            false,
            true,
            false,
            0,
        );
        let source_f16 = generate_matmul_shader(&key_f16);
        assert_parses("matmul f16", &source_f16);

        let key_q8 = PipelineKey::new(
            ScalarType::QuantizedI8,
            TileConfig::new(8, 32, 16),
            false,
            false,
            false,
            0,
        );
        let source_q8 = generate_matmul_shader(&key_q8);
        assert_parses("matmul quantized i8", &source_q8);
    }

    #[test]
    fn matmul_shader_keeps_edge_tile_invocations_until_final_write() {
        let key = PipelineKey::new(
            ScalarType::F32,
            TileConfig::new(8, 8, 8),
            false,
            false,
            false,
            0,
        );
        let source = generate_matmul_shader(&key);

        assert!(source
            .contains("let in_bounds = global_row < params.rows && global_col < params.cols;"));
        assert!(source.contains("if (in_bounds)"));
        assert!(!source.contains("return;"));
    }

    #[test]
    fn matmul_candidate_tiles_non_empty() {
        let candidates = candidate_tiles(256, 8, 256, &[]);
        assert!(!candidates.is_empty());
        assert!(candidates.contains(&TileConfig::new(16, 16, 16)));
    }

    #[test]
    fn prepacked_f32_upload_matches_shader_row_major_contract() {
        let rhs = Tensor::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let packed = PackedB::from_tensor(&rhs, Tile::col_major()).unwrap();
        let (values, total) = pack_rhs_f32(&packed, TileConfig::new(8, 8, 8));

        assert_eq!(total, rhs.data().len());
        assert_eq!(values, rhs.data());
    }

    #[test]
    fn prepacked_int8_upload_matches_shader_column_packed_contract() {
        let rhs = Tensor::from_vec(
            5,
            2,
            vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0],
        )
        .unwrap();
        let packed = PackedB::from_tensor(&rhs, Tile::col_major()).unwrap();
        let (quantized, scales, total) = pack_rhs_int8(&packed, TileConfig::new(8, 8, 8));
        let stride = rhs.shape().0.div_ceil(4);

        assert_eq!(total, rhs.shape().1 * stride * 4);
        assert_eq!(quantized.len(), rhs.shape().1 * stride);
        assert_eq!(scales.len(), rhs.shape().1);

        let decode =
            |word: i32, lane: u32| -> i8 { (((word as u32) >> (lane * 8)) & 0xFF) as u8 as i8 };
        let expected =
            |value: f32, scale: f32| -> i8 { (value / scale).round().clamp(-127.0, 127.0) as i8 };

        assert_eq!(decode(quantized[0], 0), expected(1.0, scales[0]));
        assert_eq!(decode(quantized[0], 1), expected(3.0, scales[0]));
        assert_eq!(decode(quantized[0], 2), expected(5.0, scales[0]));
        assert_eq!(decode(quantized[0], 3), expected(7.0, scales[0]));
        assert_eq!(decode(quantized[1], 0), expected(9.0, scales[0]));
        assert_eq!(decode(quantized[stride], 0), expected(-2.0, scales[1]));
        assert_eq!(decode(quantized[stride], 1), expected(-4.0, scales[1]));
        assert_eq!(decode(quantized[stride], 2), expected(-6.0, scales[1]));
        assert_eq!(decode(quantized[stride], 3), expected(-8.0, scales[1]));
        assert_eq!(decode(quantized[stride + 1], 0), expected(-10.0, scales[1]));
    }

    #[test]
    fn fused_conv_shader_wgsl_is_valid() {
        let source = instantiate_tile_template(FUSED_CONV_WGSL_TEMPLATE, TileConfig::new(8, 8, 8));
        assert_parses("fused conv", &source);
    }

    #[test]
    fn fused_conv_shader_keeps_edge_tile_invocations_until_final_write() {
        let source = instantiate_tile_template(FUSED_CONV_WGSL_TEMPLATE, TileConfig::new(8, 8, 8));

        assert!(source.contains("let in_bounds = row < total_rows && col < params.out_channels;"));
        assert!(source.contains("if (in_bounds)"));
        assert!(!source.contains("return;"));
    }

    #[test]
    fn fused_gelu_back_shader_wgsl_is_valid() {
        let source = instantiate_fused_gelu_back_template(
            FUSED_GELU_BACK_WGSL_TEMPLATE,
            FUSED_GELU_BACK_WG_ROWS,
            FUSED_GELU_BACK_WG_COLS,
        );
        assert_parses("fused gelu back", &source);
    }

    #[test]
    fn reduce_db_shader_wgsl_is_valid() {
        let source = instantiate_reduce_db_template(
            REDUCE_DB_WGSL_TEMPLATE,
            FUSED_GELU_BACK_WG_COLS,
            REDUCE_DB_WORKGROUP,
        );
        assert_parses("reduce db", &source);
    }

    #[test]
    fn fused_attention_shader_wgsl_is_valid() {
        let source = FUSED_ATTENTION_WGSL_TEMPLATE
            .replace("{WORKGROUP_SIZE}", &FUSED_ATTENTION_WORKGROUP.to_string())
            .replace("{MAX_HEAD_DIM}", &FUSED_ATTENTION_MAX_HEAD_DIM.to_string());
        assert_parses("fused attention", &source);
    }

    #[test]
    fn softmax_subgroup_shader_wgsl_is_valid() {
        assert_parses("row softmax subgroup", ROW_SOFTMAX_SUBGROUP_WGSL);
    }

    #[test]
    fn layer_norm_shader_wgsl_is_valid() {
        assert_parses("layer norm", LAYER_NORM_WGSL);
    }

    #[test]
    fn tensor_utils_shader_wgsl_is_valid() {
        assert_parses("tensor utils", TENSOR_UTILS_WGSL);
    }

    #[test]
    fn lstm_backward_scan_shader_wgsl_is_valid() {
        assert_parses("lstm backward scan", LSTM_BACKWARD_SCAN_WGSL);
        assert!(LSTM_BACKWARD_SCAN_WGSL
            .contains(&format!("@workgroup_size({LSTM_BACKWARD_SCAN_WORKGROUP})")));
        assert!(LSTM_BACKWARD_SCAN_WGSL.contains("@builtin(local_invocation_id)"));
    }

    #[test]
    fn lstm_backward_scan_shape_support_is_runtime_free() {
        assert!(lstm_backward_scan_shape_supported(1, 1));
        assert!(lstm_backward_scan_shape_supported(16, 32));
        assert!(!lstm_backward_scan_shape_supported(0, 1));
        assert!(!lstm_backward_scan_shape_supported(1, 0));
        assert!(!lstm_backward_scan_shape_supported(
            (u32::MAX as usize).saturating_add(1),
            1
        ));
        assert!(!lstm_backward_scan_shape_supported(
            1,
            (u32::MAX as usize).saturating_add(1)
        ));
    }

    #[test]
    fn softmax_zspace_shader_wgsl_is_valid() {
        assert_parses("softmax zspace projection", SOFTMAX_ZSPACE_WGSL);
    }

    #[test]
    fn softmax_spiral_shader_wgsl_is_valid() {
        assert_parses("softmax spiral consensus", SOFTMAX_SPIRAL_WGSL);
    }

    #[test]
    fn row_softmax_shader_wgsl_is_valid() {
        assert_parses("row softmax", ROW_SOFTMAX_WGSL);
    }
}

fn instantiate_fused_gelu_back_template(template: &str, wg_rows: u32, wg_cols: u32) -> String {
    template
        .replace("{WG_ROWS}", &wg_rows.to_string())
        .replace("{WG_COLS}", &wg_cols.to_string())
        .replace("{WG_TILE}", &wg_rows.saturating_mul(wg_cols).to_string())
}

fn instantiate_reduce_db_template(template: &str, wg_cols: u32, reduce_wg: u32) -> String {
    template
        .replace("{WG_COLS}", &wg_cols.to_string())
        .replace("{REDUCE_WG}", &reduce_wg.to_string())
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
    output_scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RowSoftmaxParams {
    rows: u32,
    cols: u32,
    in_stride: u32,
    out_stride: u32,
    chimera_tile: u32,
    chimera_stripes: u32,
    flags: u32,
    mask_stride: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LayerNormParams {
    rows: u32,
    cols: u32,
    flags: u32,
    _pad0: u32,
    epsilon: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TensorUtilParams {
    rows: u32,
    cols: u32,
    values: u32,
    flags: u32,
    scalar: f32,
    saturation: f32,
    porosity: f32,
    _pad2: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LstmBackwardScanParams {
    timesteps: u32,
    hidden_dim: u32,
    gate_width: u32,
    _pad0: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxZSpaceParams {
    rows: u32,
    cols: u32,
    stride: u32,
    _pad: u32,
    golden_ratio: f32,
    golden_angle: f32,
    min_energy: f32,
    _pad1: f32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SpiralConsensusParams {
    rows: u32,
    cols: u32,
    soft_stride: u32,
    mask_stride: u32,
    spiral_stride: u32,
    chimera_tile: u32,
    chimera_stripes: u32,
    flags: u32,
    phi: f32,
    phi_conjugate: f32,
    phi_bias: f32,
    leech_scale: f32,
    ramanujan_ratio: f32,
    inv_cols: f32,
    entropy_epsilon: f32,
    _pad: f32,
}

const LAYOUT_FLAG_CHIMERA: u32 = 1 << 0;
const SOFTMAX_FLAG_HARDMAX_ONLY: u32 = 1 << 1;
const SOFTMAX_FLAG_HARDMAX_MASK: u32 = 1 << 2;
const SOFTMAX_AUTOTUNE_REVISION: u64 = 2;
const SOFTMAX_AUTOTUNE_WARMUP: usize = 1;
const SOFTMAX_AUTOTUNE_SAMPLES: usize = 3;
const SOFTMAX_HISTORY_LIMIT: usize = 32;
const SOFTMAX_METROPOLIS_TRIES: usize = 4;
const SOFTMAX_METROPOLIS_TEMPERATURE: f64 = 3.5;
const SOFTMAX_ANNEAL_MIN_TEMP: f64 = 0.35;
const SOFTMAX_ANNEAL_MAX_TEMP: f64 = 2.75;
const SOFTMAX_ANNEAL_HISTORY: usize = 12;
const SOFTMAX_CONSENSUS_HISTORY: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
enum SoftmaxVariant {
    Workgroup,
    Subgroup,
}

impl SoftmaxVariant {
    fn as_str(&self) -> &'static str {
        match self {
            SoftmaxVariant::Workgroup => "workgroup",
            SoftmaxVariant::Subgroup => "subgroup",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "workgroup" => Some(SoftmaxVariant::Workgroup),
            "subgroup" => Some(SoftmaxVariant::Subgroup),
            _ => None,
        }
    }
}

fn make_softmax_telemetry_sample(
    rows: usize,
    cols: usize,
    elapsed_s: f64,
) -> Option<TelemetrySample> {
    if elapsed_s <= 0.0 {
        return None;
    }
    let elements = rows.checked_mul(cols)?;
    if elements == 0 {
        return None;
    }
    let elements_f64 = elements as f64;
    let flops = elements_f64 * SOFTMAX_FLOPS_PER_ELEMENT;
    let tflops = (flops / elapsed_s) / 1e12;
    let bytes = elements_f64 * SOFTMAX_BYTES_PER_ELEMENT;
    let bandwidth = (bytes / elapsed_s) / 1e9;
    let occupancy = estimate_softmax_occupancy(cols);
    Some(TelemetrySample::new(
        tflops as f32,
        bandwidth as f32,
        occupancy,
        None,
        false,
    ))
}

fn estimate_softmax_occupancy(cols: usize) -> f32 {
    if cols == 0 {
        return 0.0;
    }
    if cols >= SOFTMAX_WORKGROUP_SIZE as usize {
        1.0
    } else {
        (cols as f32 / SOFTMAX_WORKGROUP_SIZE).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SoftmaxZProjectMetrics {
    pub focus: f32,
    pub above: f32,
    pub here: f32,
    pub beneath: f32,
    pub swirl: f32,
    pub spiral_flux: f32,
    pub leech_enrichment: f32,
    pub ramanujan_ratio: f32,
    pub ramanujan_delta: f32,
    pub ramanujan_iterations: u32,
}

impl SoftmaxZProjectMetrics {
    #[allow(clippy::too_many_arguments)]
    fn new(
        focus: f32,
        above: f32,
        here: f32,
        beneath: f32,
        swirl: f32,
        flux: f32,
        leech_enrichment: f32,
        ramanujan_ratio: f32,
        ramanujan_delta: f32,
        ramanujan_iterations: u32,
    ) -> Self {
        Self {
            focus: focus.clamp(0.0, 1.0),
            above: above.clamp(0.0, 1.0),
            here: here.clamp(0.0, 1.0),
            beneath: beneath.clamp(0.0, 1.0),
            swirl: swirl.clamp(-1.0, 1.0),
            spiral_flux: flux.clamp(0.0, 1.0),
            leech_enrichment: leech_enrichment.clamp(0.0, 1.0),
            ramanujan_ratio: ramanujan_ratio.clamp(0.0, 2.0),
            ramanujan_delta: ramanujan_delta.max(0.0),
            ramanujan_iterations,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SoftmaxBayesEvidence {
    pub posterior_ms: f64,
    pub prior_ms: f64,
    pub confidence: f32,
    pub uplift_ms: f64,
    pub credible_low_ms: f64,
    pub credible_high_ms: f64,
}

impl SoftmaxBayesEvidence {
    fn new(
        posterior_ms: f64,
        prior_ms: f64,
        confidence: f32,
        credible_low_ms: f64,
        credible_high_ms: f64,
    ) -> Self {
        let confidence = confidence.clamp(0.0, 1.0);
        let (credible_low_ms, credible_high_ms) = if credible_low_ms <= credible_high_ms {
            (credible_low_ms.max(0.0), credible_high_ms)
        } else {
            (credible_high_ms.max(0.0), credible_low_ms)
        };
        Self {
            posterior_ms,
            prior_ms,
            confidence,
            uplift_ms: prior_ms - posterior_ms,
            credible_low_ms,
            credible_high_ms,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SoftmaxMetropolisEvidence {
    pub acceptance: f32,
    pub expected_ms: f64,
    pub tries: u32,
    pub proposal_focus: f32,
    pub proposal_flux: f32,
}

impl SoftmaxMetropolisEvidence {
    fn new(acceptance: f32, expected_ms: f64, tries: u32, focus: f32, flux: f32) -> Self {
        Self {
            acceptance: acceptance.clamp(0.0, 1.0),
            expected_ms: expected_ms.max(0.0),
            tries,
            proposal_focus: focus.clamp(0.0, 1.0),
            proposal_flux: flux.clamp(0.0, 1.0),
        }
    }

    fn identity(raw_ms: f64, focus: f32, flux: f32) -> Self {
        Self::new(0.0, raw_ms.max(0.0), 0, focus, flux)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SoftmaxSpiralAnnealEvidence {
    pub temperature: f32,
    pub annealed_ms: f64,
    pub exploration_mass: f32,
    pub entropy: f32,
    pub refreshes: u32,
}

impl SoftmaxSpiralAnnealEvidence {
    fn new(
        temperature: f32,
        annealed_ms: f64,
        exploration_mass: f32,
        entropy: f32,
        refreshes: u32,
    ) -> Self {
        Self {
            temperature: temperature.clamp(
                SOFTMAX_ANNEAL_MIN_TEMP as f32,
                SOFTMAX_ANNEAL_MAX_TEMP as f32,
            ),
            annealed_ms: annealed_ms.max(0.0),
            exploration_mass: exploration_mass.clamp(0.0, 1.0),
            entropy: entropy.clamp(0.0, 1.0),
            refreshes,
        }
    }

    fn identity(raw_ms: f64) -> Self {
        Self::new(SOFTMAX_ANNEAL_MIN_TEMP as f32, raw_ms.max(0.0), 0.0, 0.0, 0)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SoftmaxSpiralConsensusEvidence {
    pub consensus_ms: f64,
    pub synergy: f32,
    pub z_bias: f32,
    pub bayes_weight: f32,
    pub metropolis_weight: f32,
    pub anneal_weight: f32,
    pub harmony: f32,
}

impl SoftmaxSpiralConsensusEvidence {
    fn new(
        consensus_ms: f64,
        synergy: f32,
        z_bias: f32,
        bayes_weight: f32,
        metropolis_weight: f32,
        anneal_weight: f32,
        harmony: f32,
    ) -> Self {
        Self {
            consensus_ms: consensus_ms.max(0.0),
            synergy: synergy.clamp(0.0, 1.0),
            z_bias: z_bias.clamp(0.0, 1.0),
            bayes_weight: bayes_weight.clamp(0.0, 1.0),
            metropolis_weight: metropolis_weight.clamp(0.0, 1.0),
            anneal_weight: anneal_weight.clamp(0.0, 1.0),
            harmony: harmony.clamp(0.0, 1.0),
        }
    }

    fn identity(raw_ms: f64, focus: f32, flux: f32) -> Self {
        let synergy = ((focus + flux) * 0.5).clamp(0.0, 1.0);
        let z_bias = ((focus + flux) * 0.5).clamp(0.0, 1.0);
        let harmony = ((focus + flux) * 0.5).clamp(0.0, 1.0);
        Self::new(raw_ms.max(0.0), synergy, z_bias, 0.0, 0.0, 0.0, harmony)
    }
}

fn zspace_affinity(
    reference: Option<&SoftmaxZProjectMetrics>,
    proposal: &SoftmaxZProjectMetrics,
    same_variant: bool,
) -> f32 {
    let base = if let Some(current) = reference {
        let focus = 1.0 - (current.focus - proposal.focus).abs();
        let flux = 1.0 - (current.spiral_flux - proposal.spiral_flux).abs();
        let swirl_delta = (current.swirl - proposal.swirl).abs();
        let swirl = 1.0 - (swirl_delta * 0.5).min(1.0);
        let ratio = 1.0
            - (current.ramanujan_ratio - proposal.ramanujan_ratio)
                .abs()
                .min(1.0);
        let leech = 1.0
            - (current.leech_enrichment - proposal.leech_enrichment)
                .abs()
                .min(1.0);
        ((focus + flux + swirl + ratio + leech) / 5.0).clamp(0.0, 1.0)
    } else {
        let ratio = 1.0 - (proposal.ramanujan_ratio - 1.0).abs().min(1.0);
        let leech = proposal.leech_enrichment;
        ((proposal.focus + proposal.spiral_flux + ratio + leech) * 0.25).clamp(0.0, 1.0)
    };
    let bonus = if same_variant { 0.1 } else { 0.0 };
    (base + bonus).clamp(0.0, 1.0)
}

#[derive(Clone, Debug)]
struct SoftmaxSelectionRecord {
    key: String,
    variant: SoftmaxVariant,
    score_ms: f64,
    samples: usize,
    zmetrics: Option<SoftmaxZProjectMetrics>,
    bayes: Option<SoftmaxBayesEvidence>,
    metropolis: Option<SoftmaxMetropolisEvidence>,
    anneal: Option<SoftmaxSpiralAnnealEvidence>,
    consensus: Option<SoftmaxSpiralConsensusEvidence>,
}

#[derive(Clone, Copy, Debug)]
pub struct SoftmaxTelemetrySummary {
    pub avg_tflops: f32,
    pub avg_bandwidth_gbps: f32,
    pub avg_occupancy: f32,
    pub regression_rate: f32,
}

impl From<TelemetrySummary> for SoftmaxTelemetrySummary {
    fn from(summary: TelemetrySummary) -> Self {
        Self {
            avg_tflops: summary.avg_tflops,
            avg_bandwidth_gbps: summary.avg_bandwidth_gbps,
            avg_occupancy: summary.avg_occupancy,
            regression_rate: summary.regression_rate,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RoundtableBandHint {
    pub above: f32,
    pub here: f32,
    pub beneath: f32,
    pub drift: f32,
}

impl RoundtableBandHint {
    fn new(above: f32, here: f32, beneath: f32, drift: f32) -> Self {
        let above = above.max(0.0);
        let here = here.max(0.0);
        let beneath = beneath.max(0.0);
        let total = above + here + beneath;
        if total <= f32::EPSILON {
            return Self {
                above: 1.0 / 3.0,
                here: 1.0 / 3.0,
                beneath: 1.0 / 3.0,
                drift,
            };
        }
        let inv = total.recip();
        Self {
            above: (above * inv).clamp(0.0, 1.0),
            here: (here * inv).clamp(0.0, 1.0),
            beneath: (beneath * inv).clamp(0.0, 1.0),
            drift,
        }
    }

    fn from_summary(summary: SoftmaxTelemetrySummary, variant: SoftmaxVariant) -> Self {
        let occupancy = summary.avg_occupancy.clamp(0.0, 1.0);
        let regression = summary.regression_rate.clamp(0.0, 1.0);
        let variant_bias = match variant {
            SoftmaxVariant::Workgroup => 1.0,
            SoftmaxVariant::Subgroup => GOLDEN_RATIO,
        };
        let above = summary.avg_tflops.max(0.0) * variant_bias;
        let here = summary.avg_bandwidth_gbps.max(0.0) * (1.0 + 0.5 * occupancy);
        let beneath = occupancy * (1.0 + (1.0 - regression) / GOLDEN_RATIO);
        let drift = 1.0 - 2.0 * regression;
        Self::new(above, here, beneath, drift)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GoldenPulseHint {
    pub ratio_bias: f32,
    pub angle_bias_deg: f32,
    pub cooperative_weight: f32,
}

impl GoldenPulseHint {
    fn from_summary(summary: SoftmaxTelemetrySummary, variant: SoftmaxVariant) -> Self {
        let occupancy = summary.avg_occupancy.clamp(0.0, 1.0);
        let regression = summary.regression_rate.clamp(0.0, 1.0);
        let ratio = (summary.avg_tflops.max(1e-6) / summary.avg_bandwidth_gbps.max(1e-6)).ln()
            / GOLDEN_RATIO;
        let ratio_bias = ratio.clamp(-GOLDEN_RATIO, GOLDEN_RATIO);
        let variant_factor = match variant {
            SoftmaxVariant::Workgroup => 0.75,
            SoftmaxVariant::Subgroup => 1.0,
        };
        let angle_bias_deg = GOLDEN_ANGLE_DEG * occupancy * variant_factor;
        let energy = summary.avg_tflops.max(0.0) + summary.avg_bandwidth_gbps.max(0.0);
        let cooperative_weight = if energy <= f32::EPSILON {
            0.0
        } else {
            let spectral_share = (summary.avg_tflops.max(0.0) / energy).clamp(0.0, 1.0);
            (spectral_share.powf(1.0 / GOLDEN_RATIO) * (1.0 - regression)).clamp(0.0, 1.0)
        };
        Self {
            ratio_bias,
            angle_bias_deg,
            cooperative_weight,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SoftmaxZSpaceHint {
    pub focus: f32,
    pub spiral_flux: f32,
    pub roundtable: RoundtableBandHint,
    pub golden: GoldenPulseHint,
    pub leech_enrichment: f32,
    pub ramanujan_ratio: f32,
    pub ramanujan_delta: f32,
    pub ramanujan_iterations: u32,
    projection: Option<SoftmaxZProjectMetrics>,
}

impl SoftmaxZSpaceHint {
    fn from_observations(
        summary: SoftmaxTelemetrySummary,
        projection: Option<SoftmaxZProjectMetrics>,
        variant: SoftmaxVariant,
    ) -> Self {
        let occupancy = summary.avg_occupancy.clamp(0.0, 1.0);
        let regression = summary.regression_rate.clamp(0.0, 1.0);
        let harmonic = (summary.avg_tflops.max(0.0) * summary.avg_bandwidth_gbps.max(0.0)).sqrt();
        let fallback_focus = if harmonic <= f32::EPSILON {
            0.0
        } else {
            let focus_raw = harmonic * occupancy;
            (focus_raw / (focus_raw + GOLDEN_RATIO + regression)).clamp(0.0, 1.0)
        };
        let focus = projection.map(|m| m.focus).unwrap_or(fallback_focus);
        let flux = projection
            .map(|m| m.spiral_flux)
            .unwrap_or_else(|| (focus * occupancy).clamp(0.0, 1.0));
        let (leech_enrichment, ramanujan_ratio, ramanujan_delta, ramanujan_iterations) =
            if let Some(metrics) = projection {
                (
                    metrics.leech_enrichment,
                    metrics.ramanujan_ratio,
                    metrics.ramanujan_delta,
                    metrics.ramanujan_iterations,
                )
            } else {
                (0.0, 1.0, 0.0, 0)
            };
        Self {
            focus,
            spiral_flux: flux,
            roundtable: RoundtableBandHint::from_summary(summary, variant),
            golden: GoldenPulseHint::from_summary(summary, variant),
            leech_enrichment,
            ramanujan_ratio,
            ramanujan_delta,
            ramanujan_iterations,
            projection,
        }
    }

    fn from_projection(projection: SoftmaxZProjectMetrics, variant: SoftmaxVariant) -> Self {
        let pseudo = SoftmaxTelemetrySummary {
            avg_tflops: projection.focus,
            avg_bandwidth_gbps: projection.here,
            avg_occupancy: (projection.above + projection.here + projection.beneath)
                .clamp(0.0, 1.0),
            regression_rate: (1.0 - projection.focus).clamp(0.0, 1.0),
        };
        Self {
            focus: projection.focus,
            spiral_flux: projection.spiral_flux,
            roundtable: RoundtableBandHint::from_summary(pseudo, variant),
            golden: GoldenPulseHint::from_summary(pseudo, variant),
            leech_enrichment: projection.leech_enrichment,
            ramanujan_ratio: projection.ramanujan_ratio,
            ramanujan_delta: projection.ramanujan_delta,
            ramanujan_iterations: projection.ramanujan_iterations,
            projection: Some(projection),
        }
    }

    pub fn projection(&self) -> Option<&SoftmaxZProjectMetrics> {
        self.projection.as_ref()
    }
}

#[derive(Clone, Debug)]
pub struct SoftmaxSelectionSnapshot {
    pub key: String,
    variant: SoftmaxVariant,
    pub score_ms: f64,
    pub samples: usize,
    pub telemetry: Option<SoftmaxTelemetrySummary>,
    pub zspace: Option<SoftmaxZSpaceHint>,
    bayesian: Option<SoftmaxBayesEvidence>,
    metropolis: Option<SoftmaxMetropolisEvidence>,
    anneal: Option<SoftmaxSpiralAnnealEvidence>,
    consensus: Option<SoftmaxSpiralConsensusEvidence>,
}

impl SoftmaxSelectionSnapshot {
    pub fn variant_name(&self) -> &'static str {
        self.variant.as_str()
    }

    pub fn telemetry(&self) -> Option<&SoftmaxTelemetrySummary> {
        self.telemetry.as_ref()
    }

    pub fn zspace(&self) -> Option<&SoftmaxZSpaceHint> {
        self.zspace.as_ref()
    }

    pub fn projection(&self) -> Option<&SoftmaxZProjectMetrics> {
        self.zspace.as_ref().and_then(|hint| hint.projection())
    }

    pub fn bayesian(&self) -> Option<&SoftmaxBayesEvidence> {
        self.bayesian.as_ref()
    }

    pub fn metropolis(&self) -> Option<&SoftmaxMetropolisEvidence> {
        self.metropolis.as_ref()
    }

    pub fn anneal(&self) -> Option<&SoftmaxSpiralAnnealEvidence> {
        self.anneal.as_ref()
    }

    pub fn consensus(&self) -> Option<&SoftmaxSpiralConsensusEvidence> {
        self.consensus.as_ref()
    }

    fn from_record(
        record: SoftmaxSelectionRecord,
        telemetry: Option<SoftmaxTelemetrySummary>,
    ) -> Self {
        let zspace = match (telemetry, record.zmetrics) {
            (Some(summary), metrics) => Some(SoftmaxZSpaceHint::from_observations(
                summary,
                metrics,
                record.variant,
            )),
            (None, Some(metrics)) => {
                Some(SoftmaxZSpaceHint::from_projection(metrics, record.variant))
            }
            (None, None) => None,
        };
        Self {
            key: record.key,
            variant: record.variant,
            score_ms: record.score_ms,
            samples: record.samples,
            telemetry,
            zspace,
            bayesian: record.bayes,
            metropolis: record.metropolis,
            anneal: record.anneal,
            consensus: record.consensus,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SoftmaxLayoutDesc {
    in_stride: u32,
    out_stride: u32,
    chimera_tile: u32,
    chimera_stripes: u32,
    flags: u32,
}

impl SoftmaxLayoutDesc {
    fn from_layout(layout: Layout, rows: usize, cols: usize) -> Result<Self, String> {
        match layout {
            Layout::RowMajor => Ok(Self {
                in_stride: cols
                    .try_into()
                    .map_err(|_| "cols exceed u32::MAX for softmax".to_string())?,
                out_stride: cols
                    .try_into()
                    .map_err(|_| "cols exceed u32::MAX for softmax".to_string())?,
                chimera_tile: 0,
                chimera_stripes: 0,
                flags: 0,
            }),
            Layout::Chimera { stripes, tile } => {
                if stripes == 0 || tile == 0 {
                    return Err("chimera layout requires positive stripes and tile".into());
                }
                let stripes_usize = stripes as usize;
                let tile_usize = tile as usize;
                if stripes_usize * tile_usize != cols {
                    return Err("chimera layout must satisfy stripes * tile == cols".into());
                }
                Ok(Self {
                    in_stride: cols
                        .try_into()
                        .map_err(|_| "cols exceed u32::MAX for softmax".to_string())?,
                    out_stride: cols
                        .try_into()
                        .map_err(|_| "cols exceed u32::MAX for softmax".to_string())?,
                    chimera_tile: tile,
                    chimera_stripes: stripes,
                    flags: LAYOUT_FLAG_CHIMERA,
                })
            }
            Layout::ColMajor | Layout::Tiled { .. } => Err(format!(
                "softmax does not support layout {:?} for {}x{} tensors",
                layout, rows, cols
            )),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct StoredSoftmaxVariant {
    variant: String,
}

struct SpiralConsensusResources {
    rows: u32,
    bind_group: BindGroup,
    pipeline: Arc<ComputePipeline>,
    spiral_buffer: Buffer,
    metrics_buffer: Buffer,
    _params_buffer: Buffer,
    ramanujan_ratio: f64,
    ramanujan_delta: f64,
}

fn softmax_cache_key(
    ctx: &GpuContext,
    rows: usize,
    cols: usize,
    layout: &SoftmaxLayoutDesc,
) -> String {
    let info = ctx.adapter_info();
    let backend = encode_component(&format!("{:?}", info.backend));
    let driver = encode_component(&info.driver);
    let driver_info = encode_component(&info.driver_info);
    let name = encode_component(&info.name);
    format!(
        "wgpu.softmax.v{SOFTMAX_AUTOTUNE_REVISION:02}|{name}|{vendor:04x}|{device:04x}|{backend}|{driver}|{driver_info}|{rows}x{cols}|flags{flags}|tile{tile}|stripes{stripes}|samples{SOFTMAX_AUTOTUNE_SAMPLES}",
        vendor = info.vendor,
        device = info.device,
        rows = rows,
        cols = cols,
        flags = layout.flags,
        tile = layout.chimera_tile,
        stripes = layout.chimera_stripes,
    )
}

#[derive(Serialize)]
struct SoftmaxAutoContext {
    rows: usize,
    cols: usize,
    layout_flags: u32,
    chimera_tile: u32,
    chimera_stripes: u32,
    has_subgroup: bool,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedGeluBackUniforms {
    b: u32,
    o: u32,
    stride: u32,
    num_wg_x: u32,
    num_wg_y: u32,
    add_dr: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceDbUniforms {
    o: u32,
    num_wg_x: u32,
    num_wg_y: u32,
    _pad0: u32,
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

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GradInputParams {
    batch: u32,
    in_channels: u32,
    input_d: u32,
    input_h: u32,
    input_w: u32,
    input_t: u32,
    kernel_d: u32,
    kernel_h: u32,
    kernel_w: u32,
    kernel_t: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    stride_t: u32,
    pad_d: i32,
    pad_h: i32,
    pad_w: i32,
    pad_t: i32,
    dilation_d: u32,
    dilation_h: u32,
    dilation_w: u32,
    dilation_t: u32,
    out_d: u32,
    out_h: u32,
    out_w: u32,
    out_t: u32,
    out_channels: u32,
    span: u32,
    dims: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RamanujanPiParams {
    iterations: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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
    let enable_f16 = "";
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
        let signed_val = bitcast<i32>(byte_val << 24u) >> 24;
        let scale = scales[col];
        return f32(signed_val) * scale;"
            .to_string(),
    };

    let fma_line = "acc = acc + lhs_val * rhs_val;";

    let tile_mk = (key.tile_m as u64 * key.tile_k as u64) as u32;
    let tile_nk = (key.tile_n as u64 * key.tile_k as u64) as u32;

    MATMUL_WGSL_TEMPLATE
        .replace("{f16_enable}", enable_f16)
        .replace("{rhs_storage_type}", &rhs_storage_type)
        .replace("{rhs_load_body}", &rhs_load_body)
        .replace("{tile_m}", &key.tile_m.to_string())
        .replace("{tile_n}", &key.tile_n.to_string())
        .replace("{tile_k}", &key.tile_k.to_string())
        .replace("{tile_mk}", &(tile_mk.to_string() + "u"))
        .replace("{tile_nk}", &(tile_nk.to_string() + "u"))
        .replace("{workgroup_size_x}", &key.tile_n.to_string())
        .replace("{workgroup_size_y}", &key.tile_m.to_string())
        .replace("{fma_line}", fma_line)
}

fn create_wgsl_module(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> Result<wgpu::ShaderModule, anyhow::Error> {
    catch_unwind(AssertUnwindSafe(|| {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }))
    .map_err(|payload| {
        anyhow!(
            "WGSL parse error ({label}): {}",
            panic_payload_to_string(payload)
        )
    })
}

fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: Option<&wgpu::PipelineLayout>,
    module: &wgpu::ShaderModule,
    entry_point: &str,
) -> Result<wgpu::ComputePipeline, anyhow::Error> {
    catch_unwind(AssertUnwindSafe(|| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout,
            module,
            entry_point,
            compilation_options: Default::default(),
        })
    }))
    .map_err(|payload| {
        anyhow!(
            "WGPU pipeline error ({label}): {}",
            panic_payload_to_string(payload)
        )
    })
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

fn should_quantize(inner: usize, cols: usize) -> bool {
    inner.saturating_mul(cols) >= QUANTIZATION_MIN_VOLUME
}

struct QuantizedWeights {
    packed: Vec<u32>,
    scales: Vec<f32>,
}

impl QuantizedWeights {
    fn from_f32(weights: &[f32], inner: usize, cols: usize) -> Self {
        let stride = inner.div_ceil(4);
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

    fn tile(&self) -> TileConfig {
        self.tile
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

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
fn dispatch_matmul(
    ctx: &GpuContext,
    encoder: &mut wgpu::CommandEncoder,
    lhs: &Buffer,
    rhs_buffer: &Buffer,
    rhs_dtype: ScalarType,
    rhs_scales: Option<&Buffer>,
    out: &Buffer,
    rows: usize,
    inner: usize,
    cols: usize,
    tile: TileConfig,
    use_bias: bool,
    fused_ops_mask: u32,
    bias: Option<&Buffer>,
    residual: Option<&Buffer>,
    lhs_transpose: bool,
    output_scale: f32,
) -> Result<(), String> {
    let use_f16 = ctx.shader_f16;
    let key = PipelineKey::new(
        rhs_dtype,
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
    if matches!(rhs_dtype, ScalarType::QuantizedI8) {
        flags |= FLAG_USE_INT8;
    }
    if use_f16 {
        flags |= FLAG_USE_F16;
    }
    if lhs_transpose {
        flags |= FLAG_LHS_TRANSPOSE;
    }
    let params = MatmulUniforms {
        rows: rows as u32,
        cols: cols as u32,
        inner: inner as u32,
        flags,
        output_scale,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
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
        rhs_scales,
        &params_buf,
    );
    let pipeline = ctx
        .pipeline_cache
        .pipeline(key, ctx.pipeline_layout())
        .map_err(|err| err.to_string())?;
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.matmul.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        let groups_x = (cols as u32).div_ceil(tile.tile_n());
        let groups_y = (rows as u32).div_ceil(tile.tile_m());
        pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), 1);
    }
    Ok(())
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
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
) -> Result<(), String> {
    let rhs_dtype = dtype.to_scalar();
    let use_bias = bias.is_some();
    dispatch_matmul(
        ctx,
        encoder,
        lhs,
        rhs,
        rhs_dtype,
        scales,
        out,
        rows,
        inner,
        cols,
        tile,
        use_bias,
        fused_ops_mask,
        bias,
        residual,
        false,
        1.0,
    )
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
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
) -> Result<(), String> {
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
    )
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
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
) -> Result<(), String> {
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
    )
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

fn checked_tensor_util_shape(
    op: &'static str,
    input: &[f32],
    rows: usize,
    cols: usize,
) -> Result<(usize, u32, u32, u32), String> {
    if rows == 0 || cols == 0 {
        return Err(format!("{op} dimensions must be positive"));
    }
    let volume = rows
        .checked_mul(cols)
        .ok_or_else(|| format!("{op} volume exceeds usize range"))?;
    if input.len() != volume {
        return Err(format!(
            "{op} input length mismatch: expected {volume} elements, got {}",
            input.len()
        ));
    }
    let rows_u32 = u32::try_from(rows).map_err(|_| format!("{op} rows exceed u32::MAX"))?;
    let cols_u32 = u32::try_from(cols).map_err(|_| format!("{op} cols exceed u32::MAX"))?;
    let volume_u32 = u32::try_from(volume).map_err(|_| format!("{op} volume exceeds u32::MAX"))?;
    Ok((volume, rows_u32, cols_u32, volume_u32))
}

#[allow(clippy::too_many_arguments)]
fn tensor_util_internal(
    input: &[f32],
    aux: Option<&[f32]>,
    rows: usize,
    cols: usize,
    output_elements: usize,
    scalar: f32,
    saturation: f32,
    porosity: f32,
    tolerance: f32,
    pipeline: Arc<ComputePipeline>,
    dispatch_x: u32,
    label: &'static str,
) -> Result<Vec<f32>, String> {
    let (volume, rows_u32, cols_u32, volume_u32) =
        checked_tensor_util_shape(label, input, rows, cols)?;
    if output_elements == 0 {
        return Err(format!("{label} output dimensions must be positive"));
    }
    if !scalar.is_finite()
        || !saturation.is_finite()
        || !porosity.is_finite()
        || !tolerance.is_finite()
    {
        return Err(format!("{label} scalar parameters must be finite"));
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();
    let input_buf = upload_lhs(
        device,
        &format!("st.tensor.wgpu_dense.{label}.input"),
        input,
    );
    let aux_buf =
        aux.map(|data| upload_lhs(device, &format!("st.tensor.wgpu_dense.{label}.aux"), data));
    let zero_buf = if aux_buf.is_none() {
        Some(ctx.zero_storage_buffer())
    } else {
        None
    };
    let aux_binding: &Buffer = match aux_buf.as_ref() {
        Some(buf) => buf,
        None => zero_buf.as_ref().expect("zero buffer").as_ref(),
    };
    let output_buf = allocate_output(
        device,
        &format!("st.tensor.wgpu_dense.{label}.output"),
        output_elements,
    );
    let params = TensorUtilParams {
        rows: rows_u32,
        cols: cols_u32,
        values: volume_u32,
        flags: 0,
        scalar,
        saturation,
        porosity,
        _pad2: tolerance,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.params")),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.bind_group")),
        layout: &ctx.tensor_util_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: aux_binding.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.encoder")),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("st.tensor.wgpu_dense.{label}.pass")),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    debug_assert_eq!(volume, input.len());
    readback_f32(device, queue, &output_buf, output_elements)
}

#[allow(clippy::too_many_arguments)]
fn max_pool2d_util_internal(
    input: &[f32],
    aux: &[f32],
    batch: usize,
    input_cols: usize,
    work_items: usize,
    output_elements: usize,
    pipeline: Arc<ComputePipeline>,
    label: &'static str,
) -> Result<Vec<f32>, String> {
    let expected = batch
        .checked_mul(input_cols)
        .ok_or_else(|| format!("{label} input volume exceeds usize range"))?;
    if batch == 0 || input_cols == 0 || work_items == 0 || output_elements == 0 {
        return Err(format!("{label} dimensions must be positive"));
    }
    if input.len() != expected {
        return Err(format!(
            "{label} input length mismatch: expected {expected} elements, got {}",
            input.len()
        ));
    }
    let rows_u32 = u32::try_from(batch).map_err(|_| format!("{label} batch exceeds u32::MAX"))?;
    let cols_u32 =
        u32::try_from(input_cols).map_err(|_| format!("{label} input cols exceed u32::MAX"))?;
    let values_u32 =
        u32::try_from(work_items).map_err(|_| format!("{label} work items exceed u32::MAX"))?;
    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();
    let input_buf = upload_lhs(
        device,
        &format!("st.tensor.wgpu_dense.{label}.input"),
        input,
    );
    let aux_buf = upload_lhs(device, &format!("st.tensor.wgpu_dense.{label}.aux"), aux);
    let output_buf = allocate_output(
        device,
        &format!("st.tensor.wgpu_dense.{label}.output"),
        output_elements,
    );
    let params = TensorUtilParams {
        rows: rows_u32,
        cols: cols_u32,
        values: values_u32,
        flags: 0,
        scalar: 0.0,
        saturation: 0.0,
        porosity: 0.0,
        _pad2: 0.0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.params")),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.bind_group")),
        layout: &ctx.tensor_util_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: aux_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.encoder")),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("st.tensor.wgpu_dense.{label}.pass")),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(values_u32.div_ceil(TENSOR_UTIL_WORKGROUP), 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &output_buf, output_elements)
}

#[allow(clippy::too_many_arguments)]
fn embedding_util_internal(
    indices: &[f32],
    aux: &[f32],
    tokens: usize,
    embed_dim: usize,
    vocab_size: usize,
    output_elements: usize,
    pipeline: Arc<ComputePipeline>,
    label: &'static str,
) -> Result<Vec<f32>, String> {
    if tokens == 0 || embed_dim == 0 || vocab_size == 0 || output_elements == 0 {
        return Err(format!("{label} dimensions must be positive"));
    }
    if indices.len() != tokens {
        return Err(format!(
            "{label} index length mismatch: expected {tokens} elements, got {}",
            indices.len()
        ));
    }
    for &index in indices {
        if !index.is_finite() || index < 0.0 || index >= vocab_size as f32 {
            return Err(format!("{label} indices must be finite and pre-clamped"));
        }
    }
    let tokens_u32 =
        u32::try_from(tokens).map_err(|_| format!("{label} tokens exceed u32::MAX"))?;
    let embed_dim_u32 =
        u32::try_from(embed_dim).map_err(|_| format!("{label} embed_dim exceed u32::MAX"))?;
    let vocab_size_u32 =
        u32::try_from(vocab_size).map_err(|_| format!("{label} vocab_size exceed u32::MAX"))?;
    let output_u32 = u32::try_from(output_elements)
        .map_err(|_| format!("{label} output volume exceeds u32::MAX"))?;

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();
    let index_buf = upload_lhs(
        device,
        &format!("st.tensor.wgpu_dense.{label}.indices"),
        indices,
    );
    let aux_buf = upload_lhs(device, &format!("st.tensor.wgpu_dense.{label}.aux"), aux);
    let output_buf = allocate_output(
        device,
        &format!("st.tensor.wgpu_dense.{label}.output"),
        output_elements,
    );
    let params = TensorUtilParams {
        rows: tokens_u32,
        cols: embed_dim_u32,
        values: output_u32,
        flags: vocab_size_u32,
        scalar: 0.0,
        saturation: 0.0,
        porosity: 0.0,
        _pad2: 0.0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.params")),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.bind_group")),
        layout: &ctx.tensor_util_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: index_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: aux_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.encoder")),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("st.tensor.wgpu_dense.{label}.pass")),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(output_u32.div_ceil(TENSOR_UTIL_WORKGROUP), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    readback_f32(device, queue, &output_buf, output_elements)
}

fn sequence_last_step_util_internal(
    input: &[f32],
    batch: usize,
    out_steps: usize,
    features: usize,
    output_elements: usize,
    work_items: usize,
    pipeline: Arc<ComputePipeline>,
    label: &'static str,
) -> Result<Vec<f32>, String> {
    if batch == 0 || out_steps == 0 || features == 0 || output_elements == 0 || work_items == 0 {
        return Err(format!("{label} dimensions must be positive"));
    }
    let batch_u32 = u32::try_from(batch).map_err(|_| format!("{label} batch exceeds u32::MAX"))?;
    let features_u32 =
        u32::try_from(features).map_err(|_| format!("{label} features exceed u32::MAX"))?;
    let work_items_u32 =
        u32::try_from(work_items).map_err(|_| format!("{label} work items exceed u32::MAX"))?;
    let out_steps_u32 =
        u32::try_from(out_steps).map_err(|_| format!("{label} out_steps exceed u32::MAX"))?;

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();
    let input_buf = upload_lhs(
        device,
        &format!("st.tensor.wgpu_dense.{label}.input"),
        input,
    );
    let output_buf = allocate_output(
        device,
        &format!("st.tensor.wgpu_dense.{label}.output"),
        output_elements,
    );
    let params = TensorUtilParams {
        rows: batch_u32,
        cols: features_u32,
        values: work_items_u32,
        flags: out_steps_u32,
        scalar: 0.0,
        saturation: 0.0,
        porosity: 0.0,
        _pad2: 0.0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.params")),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.bind_group")),
        layout: &ctx.tensor_util_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ctx.zero_storage_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("st.tensor.wgpu_dense.{label}.encoder")),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("st.tensor.wgpu_dense.{label}.pass")),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(work_items_u32.div_ceil(TENSOR_UTIL_WORKGROUP), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    readback_f32(device, queue, &output_buf, output_elements)
}

#[allow(clippy::too_many_arguments)]
fn zspace_coherence_scan_forward_internal(
    input: &[f32],
    batch: usize,
    steps: usize,
    dim: usize,
    memory: usize,
    output_elements: usize,
    curvature: f32,
    temperature: f32,
    self_score_scale: f32,
    query_residual_scale: f32,
    pipeline: Arc<ComputePipeline>,
) -> Result<Vec<f32>, String> {
    if batch == 0 || steps == 0 || dim == 0 || memory == 0 || output_elements == 0 {
        return Err("zspace_coherence_scan_forward dimensions must be positive".into());
    }
    if memory > steps {
        return Err("zspace_coherence_scan_forward memory must not exceed steps".into());
    }
    if steps > u16::MAX as usize || memory > u16::MAX as usize {
        return Err("zspace_coherence_scan_forward steps/memory exceed packed u16 range".into());
    }
    if !curvature.is_finite()
        || curvature >= 0.0
        || !temperature.is_finite()
        || temperature <= 0.0
        || !self_score_scale.is_finite()
        || self_score_scale < 0.0
        || !query_residual_scale.is_finite()
        || query_residual_scale < 0.0
    {
        return Err("zspace_coherence_scan_forward parameters must be finite and valid".into());
    }
    let expected = batch
        .checked_mul(steps)
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| {
            "zspace_coherence_scan_forward input volume exceeds usize range".to_string()
        })?;
    if input.len() != expected {
        return Err(format!(
            "zspace_coherence_scan_forward input length mismatch: expected {expected} elements, got {}",
            input.len()
        ));
    }
    let batch_u32 = u32::try_from(batch)
        .map_err(|_| "zspace_coherence_scan_forward batch exceeds u32::MAX".to_string())?;
    let dim_u32 = u32::try_from(dim)
        .map_err(|_| "zspace_coherence_scan_forward dim exceeds u32::MAX".to_string())?;
    let output_u32 = u32::try_from(output_elements)
        .map_err(|_| "zspace_coherence_scan_forward output volume exceeds u32::MAX".to_string())?;
    let flags = (u32::try_from(steps)
        .map_err(|_| "zspace_coherence_scan_forward steps exceeds u32::MAX".to_string())?
        << 16)
        | u32::try_from(memory)
            .map_err(|_| "zspace_coherence_scan_forward memory exceeds u32::MAX".to_string())?;

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();
    let input_buf = upload_lhs(
        device,
        "st.tensor.wgpu_dense.zspace_coherence_scan_forward.input",
        input,
    );
    let output_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.zspace_coherence_scan_forward.output",
        output_elements,
    );
    let params = TensorUtilParams {
        rows: batch_u32,
        cols: dim_u32,
        values: output_u32,
        flags,
        scalar: curvature,
        saturation: temperature,
        porosity: self_score_scale,
        _pad2: query_residual_scale,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.zspace_coherence_scan_forward.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.tensor.wgpu_dense.zspace_coherence_scan_forward.bind_group"),
        layout: &ctx.tensor_util_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ctx.zero_storage_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.zspace_coherence_scan_forward.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.zspace_coherence_scan_forward.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(output_u32.div_ceil(TENSOR_UTIL_WORKGROUP), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    readback_f32(device, queue, &output_buf, output_elements)
}

#[allow(clippy::too_many_arguments)]
pub fn zspace_coherence_scan_forward(
    input: &[f32],
    batch: usize,
    steps: usize,
    dim: usize,
    memory: usize,
    curvature: f32,
    temperature: f32,
    self_score_scale: f32,
    query_residual_scale: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let context_values = batch.checked_mul(dim).ok_or_else(|| {
        "zspace_coherence_scan_forward context volume exceeds usize range".to_string()
    })?;
    let weight_values = batch.checked_mul(steps).ok_or_else(|| {
        "zspace_coherence_scan_forward weight volume exceeds usize range".to_string()
    })?;
    let output_values = context_values.checked_add(weight_values).ok_or_else(|| {
        "zspace_coherence_scan_forward output volume exceeds usize range".to_string()
    })?;
    let ctx = dense_context()?;
    let packed = zspace_coherence_scan_forward_internal(
        input,
        batch,
        steps,
        dim,
        memory,
        output_values,
        curvature,
        temperature,
        self_score_scale,
        query_residual_scale,
        ctx.tensor_util_zspace_coherence_scan_forward_pipeline
            .clone(),
    )?;
    Ok((
        packed[..context_values].to_vec(),
        packed[context_values..context_values + weight_values].to_vec(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn zspace_coherence_scan_backward(
    input: &[f32],
    grad_output: &[f32],
    weights: &[f32],
    batch: usize,
    steps: usize,
    dim: usize,
    memory: usize,
    curvature: f32,
    temperature: f32,
    self_score_scale: f32,
    query_residual_scale: f32,
) -> Result<Vec<f32>, String> {
    const LABEL: &str = "zspace_coherence_scan_backward";
    if batch == 0 || steps == 0 || dim == 0 || memory == 0 {
        return Err(format!("{LABEL} dimensions must be positive"));
    }
    if memory > steps {
        return Err(format!("{LABEL} memory must not exceed steps"));
    }
    if steps > u16::MAX as usize || memory > u16::MAX as usize {
        return Err(format!("{LABEL} steps/memory exceed packed u16 range"));
    }
    if !curvature.is_finite()
        || curvature >= 0.0
        || !temperature.is_finite()
        || temperature <= 0.0
        || !self_score_scale.is_finite()
        || self_score_scale < 0.0
        || !query_residual_scale.is_finite()
        || query_residual_scale < 0.0
    {
        return Err(format!("{LABEL} parameters must be finite and valid"));
    }
    let input_values = batch
        .checked_mul(steps)
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| format!("{LABEL} input volume exceeds usize range"))?;
    if input.len() != input_values {
        return Err(format!(
            "{LABEL} input length mismatch: expected {input_values} elements, got {}",
            input.len()
        ));
    }
    let grad_values = batch
        .checked_mul(dim)
        .ok_or_else(|| format!("{LABEL} grad output volume exceeds usize range"))?;
    if grad_output.len() != grad_values {
        return Err(format!(
            "{LABEL} grad length mismatch: expected {grad_values} elements, got {}",
            grad_output.len()
        ));
    }
    let weight_values = batch
        .checked_mul(steps)
        .ok_or_else(|| format!("{LABEL} weight volume exceeds usize range"))?;
    if weights.len() != weight_values {
        return Err(format!(
            "{LABEL} weight length mismatch: expected {weight_values} elements, got {}",
            weights.len()
        ));
    }
    let batch_u32 = u32::try_from(batch).map_err(|_| format!("{LABEL} batch exceeds u32::MAX"))?;
    let dim_u32 = u32::try_from(dim).map_err(|_| format!("{LABEL} dim exceeds u32::MAX"))?;
    let output_u32 =
        u32::try_from(input_values).map_err(|_| format!("{LABEL} volume exceeds u32::MAX"))?;
    let flags = (u32::try_from(steps).map_err(|_| format!("{LABEL} steps exceeds u32::MAX"))?
        << 16)
        | u32::try_from(memory).map_err(|_| format!("{LABEL} memory exceeds u32::MAX"))?;

    let mut aux = Vec::with_capacity(grad_values + weight_values);
    aux.extend_from_slice(grad_output);
    aux.extend_from_slice(weights);

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();
    let input_buf = upload_lhs(
        device,
        "st.tensor.wgpu_dense.zspace_coherence_scan_backward.input",
        input,
    );
    let aux_buf = upload_lhs(
        device,
        "st.tensor.wgpu_dense.zspace_coherence_scan_backward.aux",
        &aux,
    );
    let output_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.zspace_coherence_scan_backward.output",
        input_values,
    );
    let params = TensorUtilParams {
        rows: batch_u32,
        cols: dim_u32,
        values: output_u32,
        flags,
        scalar: curvature,
        saturation: temperature,
        porosity: self_score_scale,
        _pad2: query_residual_scale,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.zspace_coherence_scan_backward.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.tensor.wgpu_dense.zspace_coherence_scan_backward.bind_group"),
        layout: &ctx.tensor_util_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: aux_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.zspace_coherence_scan_backward.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.zspace_coherence_scan_backward.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(
            ctx.tensor_util_zspace_coherence_scan_backward_pipeline
                .as_ref(),
        );
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(output_u32.div_ceil(TENSOR_UTIL_WORKGROUP), 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    readback_f32(device, queue, &output_buf, input_values)
}

pub fn sequence_last_step_gather(
    input: &[f32],
    batch: usize,
    out_steps: usize,
    features: usize,
) -> Result<Vec<f32>, String> {
    if batch == 0 || out_steps == 0 || features == 0 {
        return Err("sequence_last_step_gather dimensions must be positive".into());
    }
    let input_values = batch
        .checked_mul(out_steps)
        .and_then(|value| value.checked_mul(features))
        .ok_or_else(|| "sequence_last_step_gather input volume exceeds usize range".to_string())?;
    if input.len() != input_values {
        return Err(format!(
            "sequence_last_step_gather input length mismatch: expected {input_values} elements, got {}",
            input.len()
        ));
    }
    let output_values = batch
        .checked_mul(features)
        .ok_or_else(|| "sequence_last_step_gather output volume exceeds usize range".to_string())?;
    let ctx = dense_context()?;
    sequence_last_step_util_internal(
        input,
        batch,
        out_steps,
        features,
        output_values,
        output_values,
        ctx.tensor_util_sequence_last_step_gather_pipeline.clone(),
        "sequence_last_step_gather",
    )
}

pub fn sequence_last_step_scatter(
    grad_output: &[f32],
    batch: usize,
    out_steps: usize,
    features: usize,
) -> Result<Vec<f32>, String> {
    if batch == 0 || out_steps == 0 || features == 0 {
        return Err("sequence_last_step_scatter dimensions must be positive".into());
    }
    let input_values = batch
        .checked_mul(features)
        .ok_or_else(|| "sequence_last_step_scatter input volume exceeds usize range".to_string())?;
    if grad_output.len() != input_values {
        return Err(format!(
            "sequence_last_step_scatter input length mismatch: expected {input_values} elements, got {}",
            grad_output.len()
        ));
    }
    let output_values = batch
        .checked_mul(out_steps)
        .and_then(|value| value.checked_mul(features))
        .ok_or_else(|| {
            "sequence_last_step_scatter output volume exceeds usize range".to_string()
        })?;
    let ctx = dense_context()?;
    sequence_last_step_util_internal(
        grad_output,
        batch,
        out_steps,
        features,
        output_values,
        output_values,
        ctx.tensor_util_sequence_last_step_scatter_pipeline.clone(),
        "sequence_last_step_scatter",
    )
}

pub fn scale(input: &[f32], rows: usize, cols: usize, value: f32) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("scale", input, rows, cols)?;
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        input,
        None,
        rows,
        cols,
        volume,
        value,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_scale_pipeline.clone(),
        workgroups,
        "scale",
    )
}

pub fn add(lhs: &[f32], rhs: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("add", lhs, rows, cols)?;
    if rhs.len() != lhs.len() {
        return Err(format!(
            "add rhs length mismatch: expected {} elements, got {}",
            lhs.len(),
            rhs.len()
        ));
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        lhs,
        Some(rhs),
        rows,
        cols,
        volume,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_add_pipeline.clone(),
        workgroups,
        "add",
    )
}

pub fn hadamard(lhs: &[f32], rhs: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("hadamard", lhs, rows, cols)?;
    if rhs.len() != lhs.len() {
        return Err(format!(
            "hadamard rhs length mismatch: expected {} elements, got {}",
            lhs.len(),
            rhs.len()
        ));
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        lhs,
        Some(rhs),
        rows,
        cols,
        volume,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_hadamard_pipeline.clone(),
        workgroups,
        "hadamard",
    )
}

pub fn mul_row(input: &[f32], row: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("mul_row", input, rows, cols)?;
    if row.len() != cols {
        return Err(format!(
            "mul_row row length mismatch: expected {cols} elements, got {}",
            row.len()
        ));
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        input,
        Some(row),
        rows,
        cols,
        volume,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_mul_row_pipeline.clone(),
        workgroups,
        "mul_row",
    )
}

pub fn row_affine(
    input: &[f32],
    scale_row: &[f32],
    bias_row: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("row_affine", input, rows, cols)?;
    if scale_row.len() != cols {
        return Err(format!(
            "row_affine scale length mismatch: expected {cols} elements, got {}",
            scale_row.len()
        ));
    }
    if bias_row.len() != cols {
        return Err(format!(
            "row_affine bias length mismatch: expected {cols} elements, got {}",
            bias_row.len()
        ));
    }
    let mut aux = Vec::with_capacity(cols * 2);
    aux.extend_from_slice(scale_row);
    aux.extend_from_slice(bias_row);
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        input,
        Some(aux.as_slice()),
        rows,
        cols,
        volume,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_row_affine_pipeline.clone(),
        workgroups,
        "row_affine",
    )
}

pub fn dynamic_klein_gordon_forward(
    input: &[f32],
    mass: &[f32],
    spin: &[f32],
    rows: usize,
    cols: usize,
    time_step: f32,
    damping: f32,
) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("dynamic_klein_gordon_forward", input, rows, cols)?;
    if mass.len() != cols {
        return Err(format!(
            "dynamic_klein_gordon_forward mass length mismatch: expected {cols} elements, got {}",
            mass.len()
        ));
    }
    if spin.len() != cols {
        return Err(format!(
            "dynamic_klein_gordon_forward spin length mismatch: expected {cols} elements, got {}",
            spin.len()
        ));
    }
    if !time_step.is_finite() || !damping.is_finite() {
        return Err("dynamic_klein_gordon_forward parameters must be finite".into());
    }
    let mut aux = Vec::with_capacity(cols * 2);
    aux.extend_from_slice(mass);
    aux.extend_from_slice(spin);
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        input,
        Some(aux.as_slice()),
        rows,
        cols,
        volume,
        time_step,
        damping,
        0.0,
        0.0,
        ctx.tensor_util_dynamic_klein_gordon_forward_pipeline
            .clone(),
        workgroups,
        "dynamic_klein_gordon_forward",
    )
}

pub fn dynamic_klein_gordon_backward(
    input: &[f32],
    grad_output: &[f32],
    mass: &[f32],
    spin: &[f32],
    rows: usize,
    cols: usize,
    time_step: f32,
    damping: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("dynamic_klein_gordon_backward", input, rows, cols)?;
    if grad_output.len() != volume {
        return Err(format!(
            "dynamic_klein_gordon_backward grad length mismatch: expected {volume} elements, got {}",
            grad_output.len()
        ));
    }
    if mass.len() != cols {
        return Err(format!(
            "dynamic_klein_gordon_backward mass length mismatch: expected {cols} elements, got {}",
            mass.len()
        ));
    }
    if spin.len() != cols {
        return Err(format!(
            "dynamic_klein_gordon_backward spin length mismatch: expected {cols} elements, got {}",
            spin.len()
        ));
    }
    if !time_step.is_finite() || !damping.is_finite() {
        return Err("dynamic_klein_gordon_backward parameters must be finite".into());
    }
    let mut aux = Vec::with_capacity(volume + cols * 2);
    aux.extend_from_slice(grad_output);
    aux.extend_from_slice(mass);
    aux.extend_from_slice(spin);
    let output_elements = volume
        .checked_add(cols.checked_mul(2).ok_or_else(|| {
            "dynamic_klein_gordon_backward output volume exceeds usize range".to_string()
        })?)
        .ok_or_else(|| {
            "dynamic_klein_gordon_backward output volume exceeds usize range".to_string()
        })?;
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    let packed = tensor_util_internal(
        input,
        Some(aux.as_slice()),
        rows,
        cols,
        output_elements,
        time_step,
        damping,
        0.0,
        0.0,
        ctx.tensor_util_dynamic_klein_gordon_backward_pipeline
            .clone(),
        workgroups,
        "dynamic_klein_gordon_backward",
    )?;
    let grad_input = packed[..volume].to_vec();
    let grad_mass = packed[volume..volume + cols].to_vec();
    let grad_spin = packed[volume + cols..volume + cols * 2].to_vec();
    Ok((grad_input, grad_mass, grad_spin))
}

pub fn dynamic_hamilton_jacobi_forward(
    input: &[f32],
    potential: &[f32],
    rows: usize,
    cols: usize,
    step_size: f32,
) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("dynamic_hamilton_jacobi_forward", input, rows, cols)?;
    if potential.len() != cols {
        return Err(format!(
            "dynamic_hamilton_jacobi_forward potential length mismatch: expected {cols} elements, got {}",
            potential.len()
        ));
    }
    if !step_size.is_finite() {
        return Err("dynamic_hamilton_jacobi_forward step size must be finite".into());
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        input,
        Some(potential),
        rows,
        cols,
        volume,
        step_size,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_dynamic_hamilton_jacobi_forward_pipeline
            .clone(),
        workgroups,
        "dynamic_hamilton_jacobi_forward",
    )
}

pub fn dynamic_hamilton_jacobi_backward(
    input: &[f32],
    grad_output: &[f32],
    potential: &[f32],
    rows: usize,
    cols: usize,
    step_size: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("dynamic_hamilton_jacobi_backward", input, rows, cols)?;
    if grad_output.len() != volume {
        return Err(format!(
            "dynamic_hamilton_jacobi_backward grad length mismatch: expected {volume} elements, got {}",
            grad_output.len()
        ));
    }
    if potential.len() != cols {
        return Err(format!(
            "dynamic_hamilton_jacobi_backward potential length mismatch: expected {cols} elements, got {}",
            potential.len()
        ));
    }
    if !step_size.is_finite() {
        return Err("dynamic_hamilton_jacobi_backward step size must be finite".into());
    }
    let mut aux = Vec::with_capacity(volume + cols);
    aux.extend_from_slice(grad_output);
    aux.extend_from_slice(potential);
    let output_elements = volume.checked_add(cols).ok_or_else(|| {
        "dynamic_hamilton_jacobi_backward output volume exceeds usize range".to_string()
    })?;
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    let packed = tensor_util_internal(
        input,
        Some(aux.as_slice()),
        rows,
        cols,
        output_elements,
        step_size,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_dynamic_hamilton_jacobi_backward_pipeline
            .clone(),
        workgroups,
        "dynamic_hamilton_jacobi_backward",
    )?;
    let grad_input = packed[..volume].to_vec();
    let grad_potential = packed[volume..volume + cols].to_vec();
    Ok((grad_input, grad_potential))
}

pub fn dynamic_schrodinger_forward(
    input: &[f32],
    coherence: &[f32],
    rows: usize,
    cols: usize,
    decoherence_rate: f32,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("dynamic_schrodinger_forward", input, rows, cols)?;
    if coherence.len() != cols {
        return Err(format!(
            "dynamic_schrodinger_forward coherence length mismatch: expected {cols} elements, got {}",
            coherence.len()
        ));
    }
    if !decoherence_rate.is_finite() {
        return Err("dynamic_schrodinger_forward decoherence rate must be finite".into());
    }
    let output_elements = volume.checked_mul(3).ok_or_else(|| {
        "dynamic_schrodinger_forward output volume exceeds usize range".to_string()
    })?;
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    let packed = tensor_util_internal(
        input,
        Some(coherence),
        rows,
        cols,
        output_elements,
        decoherence_rate,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_dynamic_schrodinger_forward_pipeline.clone(),
        workgroups,
        "dynamic_schrodinger_forward",
    )?;
    let interference = packed[..volume].to_vec();
    let amplitude = packed[volume..volume * 2].to_vec();
    let decoherence = packed[volume * 2..volume * 3].to_vec();
    Ok((interference, amplitude, decoherence))
}

pub fn dynamic_schrodinger_backward(
    amplitude: &[f32],
    decoherence: &[f32],
    grad_output: &[f32],
    coherence: &[f32],
    rows: usize,
    cols: usize,
    decoherence_rate: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("dynamic_schrodinger_backward", amplitude, rows, cols)?;
    if decoherence.len() != volume {
        return Err(format!(
            "dynamic_schrodinger_backward decoherence length mismatch: expected {volume} elements, got {}",
            decoherence.len()
        ));
    }
    if grad_output.len() != volume {
        return Err(format!(
            "dynamic_schrodinger_backward grad length mismatch: expected {volume} elements, got {}",
            grad_output.len()
        ));
    }
    if coherence.len() != cols {
        return Err(format!(
            "dynamic_schrodinger_backward coherence length mismatch: expected {cols} elements, got {}",
            coherence.len()
        ));
    }
    if !decoherence_rate.is_finite() {
        return Err("dynamic_schrodinger_backward decoherence rate must be finite".into());
    }
    let mut aux = Vec::with_capacity(volume * 2 + cols);
    aux.extend_from_slice(grad_output);
    aux.extend_from_slice(decoherence);
    aux.extend_from_slice(coherence);
    let output_elements = volume.checked_add(cols).ok_or_else(|| {
        "dynamic_schrodinger_backward output volume exceeds usize range".to_string()
    })?;
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    let packed = tensor_util_internal(
        amplitude,
        Some(aux.as_slice()),
        rows,
        cols,
        output_elements,
        decoherence_rate,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_dynamic_schrodinger_backward_pipeline
            .clone(),
        workgroups,
        "dynamic_schrodinger_backward",
    )?;
    let grad_input = packed[..volume].to_vec();
    let grad_coherence = packed[volume..volume + cols].to_vec();
    Ok((grad_input, grad_coherence))
}

pub fn add_scaled(
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("add_scaled", lhs, rows, cols)?;
    if rhs.len() != lhs.len() {
        return Err(format!(
            "add_scaled rhs length mismatch: expected {} elements, got {}",
            lhs.len(),
            rhs.len()
        ));
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        lhs,
        Some(rhs),
        rows,
        cols,
        volume,
        scale,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_add_scaled_pipeline.clone(),
        workgroups,
        "add_scaled",
    )
}

pub fn sub(lhs: &[f32], rhs: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("sub", lhs, rows, cols)?;
    if rhs.len() != lhs.len() {
        return Err(format!(
            "sub rhs length mismatch: expected {} elements, got {}",
            lhs.len(),
            rhs.len()
        ));
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        lhs,
        Some(rhs),
        rows,
        cols,
        volume,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_sub_pipeline.clone(),
        workgroups,
        "sub",
    )
}

pub fn transpose(input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("transpose", input, rows, cols)?;
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        input,
        None,
        rows,
        cols,
        volume,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_transpose_pipeline.clone(),
        workgroups,
        "transpose",
    )
}

pub fn embedding_gather(
    indices: &[f32],
    weights: &[f32],
    tokens: usize,
    embed_dim: usize,
) -> Result<Vec<f32>, String> {
    if embed_dim == 0 || weights.is_empty() || weights.len() % embed_dim != 0 {
        return Err("embedding_gather weights must be a non-empty vocab x embed_dim table".into());
    }
    let vocab_size = weights.len() / embed_dim;
    let ctx = dense_context()?;
    embedding_util_internal(
        indices,
        weights,
        tokens,
        embed_dim,
        vocab_size,
        tokens.saturating_mul(embed_dim),
        ctx.tensor_util_embedding_gather_pipeline.clone(),
        "embedding_gather",
    )
}

pub fn embedding_scatter_add(
    indices: &[f32],
    grad_output: &[f32],
    tokens: usize,
    vocab_size: usize,
    embed_dim: usize,
) -> Result<Vec<f32>, String> {
    if vocab_size == 0 || embed_dim == 0 {
        return Err("embedding_scatter_add vocab_size and embed_dim must be positive".into());
    }
    let expected = tokens
        .checked_mul(embed_dim)
        .ok_or_else(|| "embedding_scatter_add token volume exceeds usize range".to_string())?;
    if grad_output.len() != expected {
        return Err(format!(
            "embedding_scatter_add grad_output length mismatch: expected {expected}, got {}",
            grad_output.len()
        ));
    }
    let output_elements = vocab_size
        .checked_mul(embed_dim)
        .ok_or_else(|| "embedding_scatter_add output volume exceeds usize range".to_string())?;
    let ctx = dense_context()?;
    embedding_util_internal(
        indices,
        grad_output,
        tokens,
        embed_dim,
        vocab_size,
        output_elements,
        ctx.tensor_util_embedding_scatter_add_pipeline.clone(),
        "embedding_scatter_add",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn hypergrad_accumulate_wave(
    gradient: &[f32],
    wave: &[f32],
    rows: usize,
    cols: usize,
    curvature: f32,
    tolerance: f32,
    saturation: f32,
    porosity: f32,
) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("hypergrad_accumulate_wave", gradient, rows, cols)?;
    if wave.len() != gradient.len() {
        return Err(format!(
            "hypergrad_accumulate_wave wave length mismatch: expected {} elements, got {}",
            gradient.len(),
            wave.len()
        ));
    }
    if !curvature.is_finite()
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || !saturation.is_finite()
        || saturation <= 0.0
        || !porosity.is_finite()
    {
        return Err(
            "hypergrad_accumulate_wave parameters must be finite and positive where required"
                .into(),
        );
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        gradient,
        Some(wave),
        rows,
        cols,
        volume,
        curvature,
        saturation,
        porosity,
        tolerance,
        ctx.tensor_util_hypergrad_accumulate_wave_pipeline.clone(),
        workgroups,
        "hypergrad_accumulate_wave",
    )
}

pub fn relu(input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("relu", input, rows, cols)?;
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        input,
        None,
        rows,
        cols,
        volume,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_relu_pipeline.clone(),
        workgroups,
        "relu",
    )
}

pub fn relu_backward(
    input: &[f32],
    grad_output: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("relu_backward", input, rows, cols)?;
    if grad_output.len() != input.len() {
        return Err(format!(
            "relu_backward grad length mismatch: expected {} elements, got {}",
            input.len(),
            grad_output.len()
        ));
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        input,
        Some(grad_output),
        rows,
        cols,
        volume,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_relu_backward_pipeline.clone(),
        workgroups,
        "relu_backward",
    )
}

pub fn lstm_forward_gate_step(
    gates: &[f32],
    cell_prev: &[f32],
    hidden_dim: usize,
) -> Result<Vec<f32>, String> {
    if hidden_dim == 0 {
        return Err("lstm_forward_gate_step hidden_dim must be positive".into());
    }
    let gate_width = hidden_dim
        .checked_mul(4)
        .ok_or_else(|| "lstm_forward_gate_step gate width overflow".to_string())?;
    if gates.len() != gate_width {
        return Err(format!(
            "lstm_forward_gate_step gate length mismatch: expected {gate_width}, got {}",
            gates.len()
        ));
    }
    if cell_prev.len() != hidden_dim {
        return Err(format!(
            "lstm_forward_gate_step cell length mismatch: expected {hidden_dim}, got {}",
            cell_prev.len()
        ));
    }
    let output_elements = hidden_dim
        .checked_mul(6)
        .ok_or_else(|| "lstm_forward_gate_step output volume overflow".to_string())?;
    let hidden_dim_u32 = u32::try_from(hidden_dim)
        .map_err(|_| "lstm_forward_gate_step hidden_dim exceeds u32::MAX".to_string())?;
    let ctx = dense_context()?;
    let workgroups = hidden_dim_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        gates,
        Some(cell_prev),
        1,
        gate_width,
        output_elements,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_lstm_forward_gate_step_pipeline.clone(),
        workgroups,
        "lstm_forward_gate_step",
    )
}

#[allow(clippy::too_many_arguments)]
fn max_pool2d_config(
    label: &'static str,
    batch: usize,
    channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<Vec<f32>, String> {
    for (name, value) in [
        ("batch", batch),
        ("channels", channels),
        ("input_h", input_h),
        ("input_w", input_w),
        ("output_h", output_h),
        ("output_w", output_w),
        ("kernel_h", kernel_h),
        ("kernel_w", kernel_w),
        ("stride_h", stride_h),
        ("stride_w", stride_w),
    ] {
        if value == 0 {
            return Err(format!("{label} {name} must be positive"));
        }
        u32::try_from(value).map_err(|_| format!("{label} {name} exceeds u32::MAX"))?;
    }
    u32::try_from(pad_h).map_err(|_| format!("{label} pad_h exceeds u32::MAX"))?;
    u32::try_from(pad_w).map_err(|_| format!("{label} pad_w exceeds u32::MAX"))?;
    Ok(vec![
        channels as f32,
        input_h as f32,
        input_w as f32,
        output_h as f32,
        output_w as f32,
        kernel_h as f32,
        kernel_w as f32,
        stride_h as f32,
        stride_w as f32,
        pad_h as f32,
        pad_w as f32,
    ])
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_forward(
    input: &[f32],
    batch: usize,
    channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let input_cols = channels
        .checked_mul(input_h)
        .and_then(|value| value.checked_mul(input_w))
        .ok_or_else(|| "max_pool2d_forward input cols exceed usize range".to_string())?;
    let output_values = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_h))
        .and_then(|value| value.checked_mul(output_w))
        .ok_or_else(|| "max_pool2d_forward output volume exceeds usize range".to_string())?;
    let aux = max_pool2d_config(
        "max_pool2d_forward",
        batch,
        channels,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    )?;
    let ctx = dense_context()?;
    let packed = max_pool2d_util_internal(
        input,
        &aux,
        batch,
        input_cols,
        output_values,
        output_values.saturating_mul(2),
        ctx.tensor_util_max_pool2d_forward_pipeline.clone(),
        "max_pool2d_forward",
    )?;
    let (values, raw_indices) = packed.split_at(output_values);
    let mut indices = Vec::with_capacity(raw_indices.len());
    for &index in raw_indices {
        if !index.is_finite() || index < 0.0 || index >= input_cols as f32 {
            return Err("max_pool2d_forward returned invalid argmax index".into());
        }
        indices.push(index as usize);
    }
    Ok((values.to_vec(), indices))
}

#[allow(clippy::too_many_arguments)]
pub fn max_pool2d_backward(
    input: &[f32],
    grad_output: &[f32],
    batch: usize,
    channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<Vec<f32>, String> {
    let input_cols = channels
        .checked_mul(input_h)
        .and_then(|value| value.checked_mul(input_w))
        .ok_or_else(|| "max_pool2d_backward input cols exceed usize range".to_string())?;
    let input_values = batch
        .checked_mul(input_cols)
        .ok_or_else(|| "max_pool2d_backward input volume exceeds usize range".to_string())?;
    let output_values = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_h))
        .and_then(|value| value.checked_mul(output_w))
        .ok_or_else(|| "max_pool2d_backward output volume exceeds usize range".to_string())?;
    if grad_output.len() != output_values {
        return Err(format!(
            "max_pool2d_backward grad length mismatch: expected {output_values} elements, got {}",
            grad_output.len()
        ));
    }
    let mut aux = max_pool2d_config(
        "max_pool2d_backward",
        batch,
        channels,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    )?;
    aux.extend_from_slice(grad_output);
    let ctx = dense_context()?;
    max_pool2d_util_internal(
        input,
        &aux,
        batch,
        input_cols,
        input_values,
        input_values,
        ctx.tensor_util_max_pool2d_backward_pipeline.clone(),
        "max_pool2d_backward",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn avg_pool2d_forward(
    input: &[f32],
    batch: usize,
    channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<Vec<f32>, String> {
    let input_cols = channels
        .checked_mul(input_h)
        .and_then(|value| value.checked_mul(input_w))
        .ok_or_else(|| "avg_pool2d_forward input cols exceed usize range".to_string())?;
    let output_values = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_h))
        .and_then(|value| value.checked_mul(output_w))
        .ok_or_else(|| "avg_pool2d_forward output volume exceeds usize range".to_string())?;
    let aux = max_pool2d_config(
        "avg_pool2d_forward",
        batch,
        channels,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    )?;
    let ctx = dense_context()?;
    max_pool2d_util_internal(
        input,
        &aux,
        batch,
        input_cols,
        output_values,
        output_values,
        ctx.tensor_util_avg_pool2d_forward_pipeline.clone(),
        "avg_pool2d_forward",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn avg_pool2d_backward(
    input: &[f32],
    grad_output: &[f32],
    batch: usize,
    channels: usize,
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<Vec<f32>, String> {
    let input_cols = channels
        .checked_mul(input_h)
        .and_then(|value| value.checked_mul(input_w))
        .ok_or_else(|| "avg_pool2d_backward input cols exceed usize range".to_string())?;
    let input_values = batch
        .checked_mul(input_cols)
        .ok_or_else(|| "avg_pool2d_backward input volume exceeds usize range".to_string())?;
    let output_values = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_h))
        .and_then(|value| value.checked_mul(output_w))
        .ok_or_else(|| "avg_pool2d_backward output volume exceeds usize range".to_string())?;
    if grad_output.len() != output_values {
        return Err(format!(
            "avg_pool2d_backward grad length mismatch: expected {output_values} elements, got {}",
            grad_output.len()
        ));
    }
    let mut aux = max_pool2d_config(
        "avg_pool2d_backward",
        batch,
        channels,
        input_h,
        input_w,
        output_h,
        output_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
    )?;
    aux.extend_from_slice(grad_output);
    let ctx = dense_context()?;
    max_pool2d_util_internal(
        input,
        &aux,
        batch,
        input_cols,
        input_values,
        input_values,
        ctx.tensor_util_avg_pool2d_backward_pipeline.clone(),
        "avg_pool2d_backward",
    )
}

pub fn sum_squares(input: &[f32], rows: usize, cols: usize) -> Result<f32, String> {
    let (volume, _rows_u32, _cols_u32, _volume_u32) =
        checked_tensor_util_shape("sum_squares", input, rows, cols)?;
    let ctx = dense_context()?;
    let buffer = tensor_util_internal(
        input,
        None,
        rows,
        cols,
        1,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_sum_squares_pipeline.clone(),
        1,
        "sum_squares",
    )?;
    debug_assert_eq!(volume, input.len());
    buffer
        .first()
        .copied()
        .ok_or_else(|| "sum_squares returned no result".to_string())
}

pub fn sum_abs(input: &[f32], rows: usize, cols: usize) -> Result<f32, String> {
    let (volume, _rows_u32, _cols_u32, _volume_u32) =
        checked_tensor_util_shape("sum_abs", input, rows, cols)?;
    let ctx = dense_context()?;
    let buffer = tensor_util_internal(
        input,
        None,
        rows,
        cols,
        1,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_sum_abs_pipeline.clone(),
        1,
        "sum_abs",
    )?;
    debug_assert_eq!(volume, input.len());
    buffer
        .first()
        .copied()
        .ok_or_else(|| "sum_abs returned no result".to_string())
}

pub fn mse_loss_forward(
    prediction: &[f32],
    target: &[f32],
    rows: usize,
    cols: usize,
) -> Result<f32, String> {
    let _ = checked_tensor_util_shape("mse_loss_forward", prediction, rows, cols)?;
    if target.len() != prediction.len() {
        return Err(format!(
            "mse_loss_forward target length mismatch: expected {} elements, got {}",
            prediction.len(),
            target.len()
        ));
    }
    let ctx = dense_context()?;
    let buffer = tensor_util_internal(
        prediction,
        Some(target),
        rows,
        cols,
        1,
        1.0 / prediction.len() as f32,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_mse_loss_forward_pipeline.clone(),
        1,
        "mse_loss_forward",
    )?;
    buffer
        .first()
        .copied()
        .ok_or_else(|| "mse_loss_forward returned no result".to_string())
}

pub fn mse_loss_backward(
    prediction: &[f32],
    target: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("mse_loss_backward", prediction, rows, cols)?;
    if target.len() != prediction.len() {
        return Err(format!(
            "mse_loss_backward target length mismatch: expected {} elements, got {}",
            prediction.len(),
            target.len()
        ));
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        prediction,
        Some(target),
        rows,
        cols,
        volume,
        2.0 / prediction.len() as f32,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_mse_loss_backward_pipeline.clone(),
        workgroups,
        "mse_loss_backward",
    )
}

pub fn categorical_cross_entropy_forward(
    prediction: &[f32],
    target: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<f32, String> {
    let _ = checked_tensor_util_shape("categorical_cross_entropy_forward", prediction, rows, cols)?;
    if target.len() != prediction.len() {
        return Err(format!(
            "categorical_cross_entropy_forward target length mismatch: expected {} elements, got {}",
            prediction.len(),
            target.len()
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err("categorical_cross_entropy_forward epsilon must be finite and positive".into());
    }
    let ctx = dense_context()?;
    let buffer = tensor_util_internal(
        prediction,
        Some(target),
        rows,
        cols,
        1,
        epsilon,
        1.0 / rows as f32,
        0.0,
        0.0,
        ctx.tensor_util_categorical_cross_entropy_forward_pipeline
            .clone(),
        1,
        "categorical_cross_entropy_forward",
    )?;
    buffer
        .first()
        .copied()
        .ok_or_else(|| "categorical_cross_entropy_forward returned no result".to_string())
}

pub fn categorical_cross_entropy_backward(
    prediction: &[f32],
    target: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("categorical_cross_entropy_backward", prediction, rows, cols)?;
    if target.len() != prediction.len() {
        return Err(format!(
            "categorical_cross_entropy_backward target length mismatch: expected {} elements, got {}",
            prediction.len(),
            target.len()
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(
            "categorical_cross_entropy_backward epsilon must be finite and positive".into(),
        );
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        prediction,
        Some(target),
        rows,
        cols,
        volume,
        epsilon,
        1.0 / rows as f32,
        0.0,
        0.0,
        ctx.tensor_util_categorical_cross_entropy_backward_pipeline
            .clone(),
        workgroups,
        "categorical_cross_entropy_backward",
    )
}

pub fn hyperbolic_cross_entropy_forward(
    prediction: &[f32],
    target: &[f32],
    rows: usize,
    cols: usize,
    curvature: f32,
    epsilon: f32,
) -> Result<f32, String> {
    let _ = checked_tensor_util_shape("hyperbolic_cross_entropy_forward", prediction, rows, cols)?;
    if target.len() != prediction.len() {
        return Err(format!(
            "hyperbolic_cross_entropy_forward target length mismatch: expected {} elements, got {}",
            prediction.len(),
            target.len()
        ));
    }
    if curvature >= 0.0 || !curvature.is_finite() {
        return Err(
            "hyperbolic_cross_entropy_forward curvature must be finite and negative".into(),
        );
    }
    if !epsilon.is_finite() || epsilon <= 0.0 || epsilon >= 0.5 {
        return Err("hyperbolic_cross_entropy_forward epsilon must be in (0, 0.5)".into());
    }
    let ctx = dense_context()?;
    let buffer = tensor_util_internal(
        prediction,
        Some(target),
        rows,
        cols,
        1,
        (-curvature).sqrt(),
        epsilon,
        1.0 / prediction.len() as f32,
        0.0,
        ctx.tensor_util_hyperbolic_cross_entropy_forward_pipeline
            .clone(),
        1,
        "hyperbolic_cross_entropy_forward",
    )?;
    buffer
        .first()
        .copied()
        .ok_or_else(|| "hyperbolic_cross_entropy_forward returned no result".to_string())
}

pub fn hyperbolic_cross_entropy_backward(
    prediction: &[f32],
    target: &[f32],
    rows: usize,
    cols: usize,
    curvature: f32,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("hyperbolic_cross_entropy_backward", prediction, rows, cols)?;
    if target.len() != prediction.len() {
        return Err(format!(
            "hyperbolic_cross_entropy_backward target length mismatch: expected {} elements, got {}",
            prediction.len(),
            target.len()
        ));
    }
    if curvature >= 0.0 || !curvature.is_finite() {
        return Err(
            "hyperbolic_cross_entropy_backward curvature must be finite and negative".into(),
        );
    }
    if !epsilon.is_finite() || epsilon <= 0.0 || epsilon >= 0.5 {
        return Err("hyperbolic_cross_entropy_backward epsilon must be in (0, 0.5)".into());
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        prediction,
        Some(target),
        rows,
        cols,
        volume,
        (-curvature).sqrt(),
        epsilon,
        1.0 / prediction.len() as f32,
        0.0,
        ctx.tensor_util_hyperbolic_cross_entropy_backward_pipeline
            .clone(),
        workgroups,
        "hyperbolic_cross_entropy_backward",
    )
}

pub fn zspace_softmax_backward_fixed(
    input: &[f32],
    grad_output: &[f32],
    rows: usize,
    cols: usize,
    factor: f32,
) -> Result<Vec<f32>, String> {
    let (volume, rows_u32, _cols_u32, _volume_u32) =
        checked_tensor_util_shape("zspace_softmax_backward_fixed", input, rows, cols)?;
    if grad_output.len() != input.len() {
        return Err(format!(
            "zspace_softmax_backward_fixed grad length mismatch: expected {} elements, got {}",
            input.len(),
            grad_output.len()
        ));
    }
    if !factor.is_finite() || factor <= 0.0 {
        return Err("zspace_softmax_backward_fixed factor must be finite and positive".into());
    }
    let ctx = dense_context()?;
    tensor_util_internal(
        input,
        Some(grad_output),
        rows,
        cols,
        volume,
        factor,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_zspace_softmax_backward_fixed_pipeline
            .clone(),
        rows_u32,
        "zspace_softmax_backward_fixed",
    )
}

pub fn add_row(input: &[f32], bias: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let (volume, _rows_u32, _cols_u32, volume_u32) =
        checked_tensor_util_shape("add_row", input, rows, cols)?;
    if bias.len() != cols {
        return Err(format!(
            "add_row bias length mismatch: expected {cols} elements, got {}",
            bias.len()
        ));
    }
    let ctx = dense_context()?;
    let workgroups = volume_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    tensor_util_internal(
        input,
        Some(bias),
        rows,
        cols,
        volume,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_add_row_pipeline.clone(),
        workgroups,
        "add_row",
    )
}

pub fn sum_axis0(input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let _ = checked_tensor_util_shape("sum_axis0", input, rows, cols)?;
    let cols_u32 = u32::try_from(cols).map_err(|_| "sum_axis0 cols exceed u32::MAX".to_string())?;
    let ctx = dense_context()?;
    tensor_util_internal(
        input,
        None,
        rows,
        cols,
        cols,
        0.0,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_sum_axis0_pipeline.clone(),
        cols_u32,
        "sum_axis0",
    )
}

pub fn sum_axis0_scaled(
    input: &[f32],
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    let _ = checked_tensor_util_shape("sum_axis0_scaled", input, rows, cols)?;
    let cols_u32 =
        u32::try_from(cols).map_err(|_| "sum_axis0_scaled cols exceed u32::MAX".to_string())?;
    let ctx = dense_context()?;
    tensor_util_internal(
        input,
        None,
        rows,
        cols,
        cols,
        scale,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_sum_axis0_scaled_pipeline.clone(),
        cols_u32,
        "sum_axis0_scaled",
    )
}

pub fn project_to_poincare(
    input: &[f32],
    rows: usize,
    cols: usize,
    curvature: f32,
) -> Result<Vec<f32>, String> {
    if curvature >= 0.0 {
        return Err("project_to_poincare curvature must be negative".into());
    }
    let (volume, rows_u32, _cols_u32, _volume_u32) =
        checked_tensor_util_shape("project_to_poincare", input, rows, cols)?;
    let scale = (-curvature).sqrt();
    let ctx = dense_context()?;
    tensor_util_internal(
        input,
        None,
        rows,
        cols,
        volume,
        scale,
        0.0,
        0.0,
        0.0,
        ctx.tensor_util_project_to_poincare_pipeline.clone(),
        rows_u32,
        "project_to_poincare",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn wave_gate_project(
    input: &[f32],
    gate: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
    curvature: f32,
    saturation: f32,
    porosity: f32,
) -> Result<Vec<f32>, String> {
    if curvature >= 0.0 {
        return Err("wave_gate_project curvature must be negative".into());
    }
    if gate.len() != cols {
        return Err(format!(
            "wave_gate_project gate length mismatch: expected {cols} elements, got {}",
            gate.len()
        ));
    }
    if bias.len() != cols {
        return Err(format!(
            "wave_gate_project bias length mismatch: expected {cols} elements, got {}",
            bias.len()
        ));
    }
    let (volume, rows_u32, _cols_u32, _volume_u32) =
        checked_tensor_util_shape("wave_gate_project", input, rows, cols)?;
    let scale = (-curvature).sqrt();
    let mut aux = Vec::with_capacity(cols.saturating_mul(2));
    aux.extend_from_slice(gate);
    aux.extend_from_slice(bias);
    let ctx = dense_context()?;
    tensor_util_internal(
        input,
        Some(&aux),
        rows,
        cols,
        volume,
        scale,
        saturation,
        porosity,
        0.0,
        ctx.tensor_util_wave_gate_project_pipeline.clone(),
        rows_u32,
        "wave_gate_project",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn wave_gate_backward(
    input: &[f32],
    grad_output: &[f32],
    gate: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
    curvature: f32,
    saturation: f32,
    porosity: f32,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if curvature >= 0.0 {
        return Err("wave_gate_backward curvature must be negative".into());
    }
    if gate.len() != cols {
        return Err(format!(
            "wave_gate_backward gate length mismatch: expected {cols} elements, got {}",
            gate.len()
        ));
    }
    if bias.len() != cols {
        return Err(format!(
            "wave_gate_backward bias length mismatch: expected {cols} elements, got {}",
            bias.len()
        ));
    }
    let (volume, rows_u32, _cols_u32, _volume_u32) =
        checked_tensor_util_shape("wave_gate_backward", input, rows, cols)?;
    if grad_output.len() != volume {
        return Err(format!(
            "wave_gate_backward grad length mismatch: expected {volume} elements, got {}",
            grad_output.len()
        ));
    }
    let output_elements = volume
        .checked_mul(2)
        .ok_or_else(|| "wave_gate_backward output volume exceeds usize range".to_string())?;
    let scale = (-curvature).sqrt();
    let mut aux = Vec::with_capacity(volume.saturating_add(cols.saturating_mul(2)));
    aux.extend_from_slice(grad_output);
    aux.extend_from_slice(gate);
    aux.extend_from_slice(bias);
    let ctx = dense_context()?;
    let packed = tensor_util_internal(
        input,
        Some(&aux),
        rows,
        cols,
        output_elements,
        scale,
        saturation,
        porosity,
        f32::EPSILON,
        ctx.tensor_util_wave_gate_backward_pipeline.clone(),
        rows_u32,
        "wave_gate_backward",
    )?;
    let (grad_input, grad_affine) = packed.split_at(volume);
    Ok((grad_input.to_vec(), grad_affine.to_vec()))
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
    let tile = ctx.select_tile_config(rows, inner, cols);
    let (rhs_buffer, rhs_dtype, rhs_scales) = rhs_buf.as_binding();
    dispatch_matmul(
        &ctx,
        &mut encoder,
        &lhs_buf,
        rhs_buffer,
        rhs_dtype,
        rhs_scales,
        &out_buf,
        rows,
        inner,
        cols,
        tile,
        false,
        0,
        None,
        None,
        false,
        1.0,
    )?;
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn matmul_scaled(
    a: &[f32],
    b: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
    scale: f32,
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

    let lhs_buf = upload_lhs(device, "st.tensor.wgpu_dense.matmul_scaled.lhs", a);
    let rhs_buf = upload_weights(device, b, inner, cols);
    let out_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.matmul_scaled.out",
        rows * cols,
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.matmul_scaled.encoder"),
    });
    let tile = ctx.select_tile_config(rows, inner, cols);
    let (rhs_buffer, rhs_dtype, rhs_scales) = rhs_buf.as_binding();
    dispatch_matmul(
        &ctx,
        &mut encoder,
        &lhs_buf,
        rhs_buffer,
        rhs_dtype,
        rhs_scales,
        &out_buf,
        rows,
        inner,
        cols,
        tile,
        false,
        0,
        None,
        None,
        false,
        scale,
    )?;
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn matmul_lhs_transpose_scaled(
    lhs: &[f32],
    rhs: &[f32],
    lhs_rows: usize,
    lhs_cols: usize,
    rhs_cols: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    if lhs_rows == 0 || lhs_cols == 0 || rhs_cols == 0 {
        return Err("matrix dimensions must be positive".into());
    }
    if lhs.len() != lhs_rows * lhs_cols {
        return Err(format!(
            "lhs buffer length mismatch: expected {} elements, got {}",
            lhs_rows * lhs_cols,
            lhs.len()
        ));
    }
    if rhs.len() != lhs_rows * rhs_cols {
        return Err(format!(
            "rhs buffer length mismatch: expected {} elements, got {}",
            lhs_rows * rhs_cols,
            rhs.len()
        ));
    }

    let rows = lhs_cols;
    let inner = lhs_rows;
    let cols = rhs_cols;
    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let lhs_buf = upload_lhs(device, "st.tensor.wgpu_dense.matmul_lhs_t_scaled.lhs", lhs);
    let rhs_buf = upload_weights(device, rhs, inner, cols);
    let out_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.matmul_lhs_t_scaled.out",
        rows * cols,
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.matmul_lhs_t_scaled.encoder"),
    });
    let tile = ctx.select_tile_config(rows, inner, cols);
    let (rhs_buffer, rhs_dtype, rhs_scales) = rhs_buf.as_binding();
    dispatch_matmul(
        &ctx,
        &mut encoder,
        &lhs_buf,
        rhs_buffer,
        rhs_dtype,
        rhs_scales,
        &out_buf,
        rows,
        inner,
        cols,
        tile,
        false,
        0,
        None,
        None,
        true,
        scale,
    )?;
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
    let requested_tile = ctx.select_tile_config(rows, inner, cols);
    let weights = ctx.rhs_from_packed(packed_rhs, requested_tile)?;
    let tile = weights.tile();
    let rhs_buffers = WeightBuffers {
        buffer: weights.buffer(),
        dtype: weights.dtype(),
        scales: weights.scales(),
    };
    let (rhs_buffer, rhs_dtype, rhs_scales) = rhs_buffers.as_binding();
    dispatch_matmul(
        &ctx,
        &mut encoder,
        &lhs_buf,
        rhs_buffer,
        rhs_dtype,
        rhs_scales,
        &out_buf,
        rows,
        inner,
        cols,
        tile,
        false,
        0,
        None,
        None,
        false,
        1.0,
    )?;
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn matmul_prepacked_bias(
    lhs: &[f32],
    packed_rhs: &PackedB,
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
    if packed_rhs.inner() != inner {
        return Err("packed rhs inner dimension mismatch".into());
    }
    if packed_rhs.cols() != cols {
        return Err("packed rhs column dimension mismatch".into());
    }
    if bias.len() != cols {
        return Err(format!(
            "bias length mismatch: expected {cols} elements, got {}",
            bias.len()
        ));
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let lhs_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.prepacked_bias.lhs"),
        contents: bytemuck::cast_slice(lhs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.prepacked_bias.bias"),
        contents: bytemuck::cast_slice(bias),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.prepacked_bias.out",
        rows * cols,
    );

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.prepacked_bias.encoder"),
    });
    let requested_tile = ctx.select_tile_config(rows, inner, cols);
    let weights = ctx.rhs_from_packed(packed_rhs, requested_tile)?;
    let tile = weights.tile();
    let rhs_buffers = WeightBuffers {
        buffer: weights.buffer(),
        dtype: weights.dtype(),
        scales: weights.scales(),
    };
    let (rhs_buffer, rhs_dtype, rhs_scales) = rhs_buffers.as_binding();
    dispatch_matmul(
        &ctx,
        &mut encoder,
        &lhs_buf,
        rhs_buffer,
        rhs_dtype,
        rhs_scales,
        &out_buf,
        rows,
        inner,
        cols,
        tile,
        true,
        0,
        Some(&bias_buf),
        None,
        false,
        1.0,
    )?;
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

#[allow(clippy::too_many_arguments)]
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
    let tile = ctx.select_tile_config(rows, inner, cols);
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
    let (rhs_buffer, rhs_dtype, rhs_scales) = rhs_buf.as_binding();
    dispatch_matmul(
        &ctx,
        &mut encoder,
        &lhs_buf,
        rhs_buffer,
        rhs_dtype,
        rhs_scales,
        &out_buf,
        rows,
        inner,
        cols,
        tile,
        true,
        fused_mask,
        Some(&bias_buf),
        residual_buf.as_ref(),
        false,
        1.0,
    )?;
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &out_buf, rows * cols)
}

pub fn fused_gelu_backward(
    z: &[f32],
    grad: &[f32],
    residual_grad: Option<&[f32]>,
    rows: usize,
    cols: usize,
) -> Result<FusedGeluBackwardOutput, String> {
    if rows == 0 || cols == 0 {
        return Err("tensor dimensions must be positive".into());
    }
    let expected = rows * cols;
    if z.len() != expected {
        return Err(format!(
            "z length mismatch: expected {} elements, got {}",
            expected,
            z.len()
        ));
    }
    if grad.len() != expected {
        return Err(format!(
            "grad length mismatch: expected {} elements, got {}",
            expected,
            grad.len()
        ));
    }
    if let Some(residual) = residual_grad {
        if residual.len() != expected {
            return Err(format!(
                "residual gradient length mismatch: expected {} elements, got {}",
                expected,
                residual.len()
            ));
        }
    }

    let rows_u32 = u32::try_from(rows).map_err(|_| "rows exceed u32::MAX".to_string())?;
    let cols_u32 = u32::try_from(cols).map_err(|_| "cols exceed u32::MAX".to_string())?;

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let z_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.gelu_back.z"),
        contents: bytemuck::cast_slice(z),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let grad_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.gelu_back.grad"),
        contents: bytemuck::cast_slice(grad),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let gz_buf = allocate_output(device, "st.tensor.wgpu_dense.gelu_back.gz", expected);

    let residual_data = match residual_grad {
        Some(data) => data.to_vec(),
        None => vec![0.0; expected],
    };
    let dr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.gelu_back.dr"),
        contents: bytemuck::cast_slice(&residual_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    let num_wg_x = cols_u32.div_ceil(FUSED_GELU_BACK_WG_COLS);
    let num_wg_y = rows_u32.div_ceil(FUSED_GELU_BACK_WG_ROWS);
    let partial_len = (num_wg_x * num_wg_y * FUSED_GELU_BACK_WG_COLS) as usize;
    let partials_zero = vec![0.0f32; partial_len];
    let partials_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.gelu_back.partials"),
        contents: bytemuck::cast_slice(&partials_zero),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let db_buf = allocate_output(device, "st.tensor.wgpu_dense.gelu_back.db", cols);

    let fused_uniforms = FusedGeluBackUniforms {
        b: rows_u32,
        o: cols_u32,
        stride: cols_u32,
        num_wg_x,
        num_wg_y,
        add_dr: if residual_grad.is_some() { 1 } else { 0 },
        _pad0: 0,
        _pad1: 0,
    };
    let fused_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.gelu_back.uniforms"),
        contents: bytemuck::bytes_of(&fused_uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let reduce_uniforms = ReduceDbUniforms {
        o: cols_u32,
        num_wg_x,
        num_wg_y,
        _pad0: 0,
    };
    let reduce_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.gelu_back.reduce_uniforms"),
        contents: bytemuck::bytes_of(&reduce_uniforms),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let fused_bind_group = ctx.fused_gelu_back_bind_group(
        &z_buf,
        &grad_buf,
        &gz_buf,
        &dr_buf,
        &partials_buf,
        &fused_uniform_buf,
    );
    let reduce_bind_group = ctx.reduce_db_bind_group(&partials_buf, &db_buf, &reduce_uniform_buf);

    let fused_pipeline = ctx.fused_gelu_back_pipeline.clone();
    let reduce_pipeline = ctx.reduce_db_pipeline.clone();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.gelu_back.encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.gelu_back.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(fused_pipeline.as_ref());
        pass.set_bind_group(0, &fused_bind_group, &[]);
        pass.dispatch_workgroups(num_wg_x, num_wg_y, 1);
    }

    let reduce_groups = cols_u32.div_ceil(REDUCE_DB_WORKGROUP);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.gelu_back.reduce"),
            timestamp_writes: None,
        });
        pass.set_pipeline(reduce_pipeline.as_ref());
        pass.set_bind_group(0, &reduce_bind_group, &[]);
        pass.dispatch_workgroups(reduce_groups, 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    let gz = readback_f32(device, queue, &gz_buf, expected)?;
    let dr_out = readback_f32(device, queue, &dr_buf, expected)?;
    let db = readback_f32(device, queue, &db_buf, cols)?;

    Ok((gz, dr_out, db))
}

pub fn gelu_backward(
    z: &[f32],
    grad: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let (gz, _dr, _db) = fused_gelu_backward(z, grad, None, rows, cols)?;
    Ok(gz)
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm_affine(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    layer_norm_internal(input, None, gamma, beta, rows, cols, epsilon)
}

#[allow(clippy::too_many_arguments)]
pub fn layer_norm_affine_add(
    input: &[f32],
    residual: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    layer_norm_internal(input, Some(residual), gamma, beta, rows, cols, epsilon)
}

#[allow(clippy::too_many_arguments)]
fn layer_norm_internal(
    input: &[f32],
    residual: Option<&[f32]>,
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    if rows == 0 || cols == 0 {
        return Err("layer norm dimensions must be positive".into());
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err("layer norm epsilon must be finite and non-negative".into());
    }

    let volume = rows
        .checked_mul(cols)
        .ok_or_else(|| "layer norm volume exceeds usize range".to_string())?;

    if input.len() != volume {
        return Err(format!(
            "layer norm input length mismatch: expected {} elements, got {}",
            volume,
            input.len()
        ));
    }
    if let Some(residual) = residual {
        if residual.len() != volume {
            return Err(format!(
                "layer norm residual length mismatch: expected {} elements, got {}",
                volume,
                residual.len()
            ));
        }
    }
    if gamma.len() != cols {
        return Err(format!(
            "layer norm gamma length mismatch: expected {} elements, got {}",
            cols,
            gamma.len()
        ));
    }
    if beta.len() != cols {
        return Err(format!(
            "layer norm beta length mismatch: expected {} elements, got {}",
            cols,
            beta.len()
        ));
    }

    if rows > u32::MAX as usize || cols > u32::MAX as usize {
        return Err("layer norm dimensions exceed u32 dispatch range".into());
    }

    let rows_u32 =
        u32::try_from(rows).map_err(|_| "layer norm rows exceed u32 dispatch range".to_string())?;
    let cols_u32 =
        u32::try_from(cols).map_err(|_| "layer norm cols exceed u32 dispatch range".to_string())?;

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let input_buf = upload_lhs(device, "st.tensor.wgpu_dense.layer_norm.input", input);
    let residual_buf =
        residual.map(|data| upload_lhs(device, "st.tensor.wgpu_dense.layer_norm.residual", data));
    let gamma_buf = upload_lhs(device, "st.tensor.wgpu_dense.layer_norm.gamma", gamma);
    let beta_buf = upload_lhs(device, "st.tensor.wgpu_dense.layer_norm.beta", beta);
    let output_buf = allocate_output(device, "st.tensor.wgpu_dense.layer_norm.output", volume);

    let flags = if residual.is_some() {
        LAYER_NORM_FLAG_USE_RESIDUAL
    } else {
        0
    };

    crate::emit_tensor_op_meta("layer_norm", || {
        serde_json::json!({
            "backend": "wgpu_dense",
            "rows": rows,
            "cols": cols,
            "epsilon": epsilon,
            "flags": {
                "use_residual": (flags & LAYER_NORM_FLAG_USE_RESIDUAL) != 0,
            },
            "kernel": {
                "workgroup_size": LAYER_NORM_WORKGROUP,
            }
        })
    });

    let params = LayerNormParams {
        rows: rows_u32,
        cols: cols_u32,
        flags,
        _pad0: 0,
        epsilon,
        _pad1: 0.0,
        _pad2: 0.0,
        _pad3: 0.0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.layer_norm.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let zero_storage = ctx.zero_storage_buffer();
    let residual_binding: &Buffer = match residual_buf.as_ref() {
        Some(buf) => buf,
        None => zero_storage.as_ref(),
    };

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.tensor.wgpu_dense.layer_norm.bind_group"),
        layout: &ctx.layer_norm_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: residual_binding.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gamma_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: beta_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });

    let pipeline = ctx.layer_norm_pipeline.clone();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.layer_norm.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.layer_norm.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows_u32, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &output_buf, volume)
}

pub fn row_softmax(
    input: &[f32],
    rows: usize,
    cols: usize,
    layout: Layout,
) -> Result<Vec<f32>, String> {
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

    let layout_desc = SoftmaxLayoutDesc::from_layout(layout, rows, cols)?;
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
        in_stride: layout_desc.in_stride,
        out_stride: layout_desc.out_stride,
        chimera_tile: layout_desc.chimera_tile,
        chimera_stripes: layout_desc.chimera_stripes,
        flags: layout_desc.flags,
        mask_stride: layout_desc.out_stride,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.softmax.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.softmax_bind_group(&input_buf, &output_buf, None, &params_buf);
    let (pipeline, variant) = ctx.select_softmax_pipeline(rows, cols, &layout_desc)?;

    crate::emit_tensor_op_meta("row_softmax", || {
        serde_json::json!({
            "backend": "wgpu_dense",
            "variant": variant.as_str(),
            "rows": rows,
            "cols": cols,
            "layout": {
                "flags": layout_desc.flags,
                "chimera_tile": layout_desc.chimera_tile,
                "chimera_stripes": layout_desc.chimera_stripes,
            }
        })
    });

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

pub fn row_softmax_hardmax(
    input: &[f32],
    rows: usize,
    cols: usize,
    layout: Layout,
) -> Result<(Vec<f32>, Vec<f32>), String> {
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

    let mut layout_desc = SoftmaxLayoutDesc::from_layout(layout, rows, cols)?;
    layout_desc.flags |= SOFTMAX_FLAG_HARDMAX_MASK;

    let device = ctx.device();
    let queue = ctx.queue();

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.softmax_hardmax.input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let softmax_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.softmax_hardmax.softmax",
        rows * cols,
    );
    let mask_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.softmax_hardmax.mask",
        rows * cols,
    );
    let params = RowSoftmaxParams {
        rows: rows_u32,
        cols: cols_u32,
        in_stride: layout_desc.in_stride,
        out_stride: layout_desc.out_stride,
        chimera_tile: layout_desc.chimera_tile,
        chimera_stripes: layout_desc.chimera_stripes,
        flags: layout_desc.flags,
        mask_stride: layout_desc.out_stride,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.softmax_hardmax.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.softmax_bind_group(&input_buf, &softmax_buf, Some(&mask_buf), &params_buf);
    let (pipeline, variant) = ctx.select_softmax_pipeline(rows, cols, &layout_desc)?;

    crate::emit_tensor_op_meta("row_softmax_hardmax", || {
        serde_json::json!({
            "backend": "wgpu_dense",
            "variant": variant.as_str(),
            "rows": rows,
            "cols": cols,
            "layout": {
                "flags": layout_desc.flags,
                "chimera_tile": layout_desc.chimera_tile,
                "chimera_stripes": layout_desc.chimera_stripes,
            }
        })
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.softmax_hardmax.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_hardmax.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows_u32, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    let softmax = readback_f32(device, queue, &softmax_buf, rows * cols)?;
    let mask = readback_f32(device, queue, &mask_buf, rows * cols)?;
    Ok((softmax, mask))
}

pub fn row_softmax_hardmax_spiral(
    input: &[f32],
    rows: usize,
    cols: usize,
    layout: Layout,
) -> Result<RowSoftmaxHardmaxSpiralOutput, String> {
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

    let mut layout_desc = SoftmaxLayoutDesc::from_layout(layout, rows, cols)?;
    layout_desc.flags |= SOFTMAX_FLAG_HARDMAX_MASK;

    let device = ctx.device();
    let queue = ctx.queue();

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.softmax_spiral.input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let softmax_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.softmax_spiral.softmax",
        rows * cols,
    );
    let mask_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.softmax_spiral.mask",
        rows * cols,
    );
    let params = RowSoftmaxParams {
        rows: rows_u32,
        cols: cols_u32,
        in_stride: layout_desc.in_stride,
        out_stride: layout_desc.out_stride,
        chimera_tile: layout_desc.chimera_tile,
        chimera_stripes: layout_desc.chimera_stripes,
        flags: layout_desc.flags,
        mask_stride: layout_desc.out_stride,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.softmax_spiral.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.softmax_bind_group(&input_buf, &softmax_buf, Some(&mask_buf), &params_buf);
    let (pipeline, variant) = ctx.select_softmax_pipeline(rows, cols, &layout_desc)?;

    let consensus_resources =
        ctx.prepare_spiral_consensus(rows, cols, &layout_desc, &softmax_buf, &mask_buf);
    let spiral_consensus_gpu = consensus_resources.is_some();

    crate::emit_tensor_op_meta("row_softmax_hardmax_spiral", || {
        serde_json::json!({
            "backend": "wgpu_dense",
            "variant": variant.as_str(),
            "rows": rows,
            "cols": cols,
            "spiral_consensus_gpu": spiral_consensus_gpu,
            "layout": {
                "flags": layout_desc.flags,
                "chimera_tile": layout_desc.chimera_tile,
                "chimera_stripes": layout_desc.chimera_stripes,
            }
        })
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.softmax_spiral.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_spiral.softmax_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rows_u32, 1, 1);
    }

    if let Some(ref resources) = consensus_resources {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.softmax_spiral.consensus_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(resources.pipeline.as_ref());
        pass.set_bind_group(0, &resources.bind_group, &[]);
        pass.dispatch_workgroups(resources.rows, 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    let softmax = readback_f32(device, queue, &softmax_buf, rows * cols)?;
    let hardmax = readback_f32(device, queue, &mask_buf, rows * cols)?;

    if let Some(resources) = consensus_resources {
        let spiral = readback_f32(device, queue, &resources.spiral_buffer, rows * cols)?;
        let metrics = readback_f32(device, queue, &resources.metrics_buffer, rows * 4)?;
        let mut total_entropy = 0.0_f64;
        let mut total_hardmass = 0.0_f64;
        let mut total_enrichment = 0.0_f64;
        let mut total_coherence = 0.0_f64;
        for chunk in metrics.chunks_exact(4) {
            let entropy = if chunk[0].is_finite() {
                f64::from(chunk[0].max(0.0))
            } else {
                0.0
            };
            let hardmass = if chunk[1].is_finite() {
                f64::from(chunk[1].max(0.0))
            } else {
                0.0
            };
            let enrichment = if chunk[2].is_finite() {
                f64::from(chunk[2].max(0.0))
            } else {
                0.0
            };
            let coherence = if chunk[3].is_finite() {
                f64::from(chunk[3].clamp(0.0, 1.0))
            } else {
                0.0
            };

            total_entropy += entropy;
            total_hardmass += hardmass;
            total_enrichment += enrichment;
            total_coherence += coherence;
        }
        let rows_f64 = rows as f64;
        let inv_rows = if rows_f64 > 0.0 { 1.0 / rows_f64 } else { 0.0 };
        let stats = SpiralConsensusStats {
            phi: GOLDEN_RATIO as f64,
            phi_conjugate: GOLDEN_RATIO_CONJUGATE as f64,
            phi_bias: GOLDEN_RATIO_BIAS as f64,
            ramanujan_ratio: resources.ramanujan_ratio,
            ramanujan_delta: resources.ramanujan_delta,
            average_enrichment: total_enrichment * inv_rows,
            mean_entropy: total_entropy * inv_rows,
            mean_hardmass: total_hardmass * inv_rows,
            spiral_coherence: total_coherence * inv_rows,
        };
        Ok((softmax, hardmax, spiral, stats))
    } else {
        let (spiral, stats) = spiral_softmax_hardmax_consensus(&softmax, &hardmax, rows, cols);
        Ok((softmax, hardmax, spiral, stats))
    }
}

pub fn row_hardmax(
    input: &[f32],
    rows: usize,
    cols: usize,
    layout: Layout,
) -> Result<Vec<f32>, String> {
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

    let mut layout_desc = SoftmaxLayoutDesc::from_layout(layout, rows, cols)?;
    layout_desc.flags |= SOFTMAX_FLAG_HARDMAX_ONLY;

    let device = ctx.device();
    let queue = ctx.queue();

    let input_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.hardmax.input"),
        contents: bytemuck::cast_slice(input),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buf = allocate_output(device, "st.tensor.wgpu_dense.hardmax.output", rows * cols);
    let params = RowSoftmaxParams {
        rows: rows_u32,
        cols: cols_u32,
        in_stride: layout_desc.in_stride,
        out_stride: layout_desc.out_stride,
        chimera_tile: layout_desc.chimera_tile,
        chimera_stripes: layout_desc.chimera_stripes,
        flags: layout_desc.flags,
        mask_stride: layout_desc.out_stride,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.hardmax.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group = ctx.softmax_bind_group(&input_buf, &output_buf, None, &params_buf);
    let (pipeline, variant) = ctx.select_softmax_pipeline(rows, cols, &layout_desc)?;

    crate::emit_tensor_op_meta("row_hardmax", || {
        serde_json::json!({
            "backend": "wgpu_dense",
            "variant": variant.as_str(),
            "rows": rows,
            "cols": cols,
            "layout": {
                "flags": layout_desc.flags,
                "chimera_tile": layout_desc.chimera_tile,
                "chimera_stripes": layout_desc.chimera_stripes,
            }
        })
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.hardmax.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.hardmax.pass"),
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

pub fn supports_layer_norm(rows: usize, cols: usize) -> bool {
    if rows == 0 || cols == 0 {
        return false;
    }
    if rows > u32::MAX as usize || cols > u32::MAX as usize {
        return false;
    }
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

pub fn supports_row_softmax_hardmax(rows: usize, cols: usize) -> bool {
    supports_row_softmax(rows, cols)
}

pub fn supports_row_softmax_hardmax_spiral(rows: usize, cols: usize) -> bool {
    supports_row_softmax(rows, cols)
}

pub fn supports_row_hardmax(rows: usize, cols: usize) -> bool {
    supports_row_softmax(rows, cols)
}

pub fn softmax_autotune_snapshot() -> Option<Vec<SoftmaxSelectionSnapshot>> {
    let ctx = dense_context().ok()?;
    let history = ctx.softmax_history.lock().ok()?.clone();
    let key_map = ctx
        .softmax_telemetry_keys
        .lock()
        .ok()
        .map(|guard| guard.clone());
    let registry_guard = global_autotune_registry().lock().ok();

    let mut snapshots = Vec::with_capacity(history.len());
    for record in history.into_iter() {
        let telemetry = key_map
            .as_ref()
            .and_then(|map| map.get(&record.key))
            .and_then(|autokey| {
                registry_guard
                    .as_ref()
                    .and_then(|registry| registry.summary(autokey))
            })
            .map(SoftmaxTelemetrySummary::from);
        snapshots.push(SoftmaxSelectionSnapshot::from_record(record, telemetry));
    }

    Some(snapshots)
}

#[allow(clippy::too_many_arguments)]
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
    let z_binding = z_bias_buf.as_ref().unwrap_or_else(|| zero_storage.as_ref());
    let attn_binding = attn_bias_buf
        .as_ref()
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

    crate::emit_tensor_op_meta("scaled_dot_attention", || {
        serde_json::json!({
            "backend": "wgpu_dense",
            "contexts": contexts,
            "sequence": sequence,
            "head_dim": head_dim,
            "scale": scale,
            "flags": {
                "use_z_bias": (flags & FUSED_ATTENTION_FLAG_USE_Z_BIAS) != 0,
                "use_attn_bias": (flags & FUSED_ATTENTION_FLAG_USE_ATTN_BIAS) != 0,
            },
            "kernel": {
                "workgroup_size": FUSED_ATTENTION_WORKGROUP,
                "max_head_dim": kernel.max_head_dim,
            }
        })
    });

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

#[allow(clippy::too_many_arguments)]
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
    let zero_bias;
    let bias_data = match bias {
        Some(values) => values,
        None => {
            zero_bias = vec![0.0f32; out_channels];
            &zero_bias
        }
    };

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
    let bias_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.conv.bias"),
        contents: bytemuck::cast_slice(bias_data),
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
    let tile_config = ctx.select_tile_config(rows, span, out_channels);
    let fused_pipeline = ctx.fused_conv_pipeline_for(tile_config)?;
    let fused_bind_group = ctx.fused_conv_bind_group(
        &input_buf,
        &weight_buf,
        &output_buf,
        &bias_buf,
        &conv_params_buf,
    );
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.conv.fused_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(fused_pipeline.as_ref());
        pass.set_bind_group(0, &fused_bind_group, &[]);
        let groups_x = (out_channels as u32).div_ceil(tile_config.tile_n());
        let groups_y = (rows as u32).div_ceil(tile_config.tile_m());
        pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), 1);
    }

    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &output_buf, rows * out_channels)
}

struct GradInputLaunch<'a> {
    grad_matrix: &'a [f32],
    weights: &'a [f32],
    batch: usize,
    in_channels: usize,
    input: [usize; 4],
    kernel: [usize; 4],
    stride: [usize; 4],
    pad: [i32; 4],
    dilation: [usize; 4],
    output: [usize; 4],
    out_channels: usize,
    dims: u32,
}

fn conv_grad_input_fused_common(config: GradInputLaunch<'_>) -> Result<Vec<f32>, String> {
    let GradInputLaunch {
        grad_matrix,
        weights,
        batch,
        in_channels,
        input,
        kernel,
        stride,
        pad,
        dilation,
        output,
        out_channels,
        dims,
    } = config;

    if batch == 0
        || in_channels == 0
        || kernel.iter().product::<usize>() == 0
        || out_channels == 0
        || output.iter().product::<usize>() == 0
    {
        return Err("convolution dimensions must be positive".into());
    }

    let rows = batch
        .checked_mul(output.iter().product::<usize>())
        .ok_or_else(|| "output spatial overflow".to_string())?;
    let span = in_channels
        .checked_mul(kernel.iter().product::<usize>())
        .ok_or_else(|| "kernel span overflow".to_string())?;

    if grad_matrix.len() != rows * out_channels {
        return Err("grad matrix length mismatch".into());
    }
    if weights.len() != out_channels * span {
        return Err("weight buffer length mismatch".into());
    }

    let total_bc = batch
        .checked_mul(in_channels)
        .and_then(|value| value.checked_mul(input[0]))
        .and_then(|value| value.checked_mul(input[3]))
        .ok_or_else(|| "input channel volume overflow".to_string())?;
    if total_bc > u32::MAX as usize {
        return Err("input channel volume exceeds GPU limits".into());
    }
    let input_volume = total_bc
        .checked_mul(input[1])
        .and_then(|value| value.checked_mul(input[2]))
        .ok_or_else(|| "input tensor volume overflow".to_string())?;

    let exceeds_limit = [
        rows,
        span,
        batch,
        in_channels,
        input[0],
        input[1],
        input[2],
        input[3],
        kernel[0],
        kernel[1],
        kernel[2],
        kernel[3],
        stride[0],
        stride[1],
        stride[2],
        stride[3],
        dilation[0],
        dilation[1],
        dilation[2],
        dilation[3],
        output[0],
        output[1],
        output[2],
        output[3],
        out_channels,
    ]
    .into_iter()
    .any(|value| value > u32::MAX as usize);
    if exceeds_limit {
        return Err("convolution dimensions exceed GPU limits".into());
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let grad_matrix_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.conv.grad_matrix"),
        contents: bytemuck::cast_slice(grad_matrix),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let weights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.conv.grad_weights"),
        contents: bytemuck::cast_slice(weights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_buf = allocate_output(device, "st.tensor.wgpu_dense.conv.grad_input", input_volume);

    let params = GradInputParams {
        batch: batch as u32,
        in_channels: in_channels as u32,
        input_d: input[0] as u32,
        input_h: input[1] as u32,
        input_w: input[2] as u32,
        input_t: input[3] as u32,
        kernel_d: kernel[0] as u32,
        kernel_h: kernel[1] as u32,
        kernel_w: kernel[2] as u32,
        kernel_t: kernel[3] as u32,
        stride_d: stride[0] as u32,
        stride_h: stride[1] as u32,
        stride_w: stride[2] as u32,
        stride_t: stride[3] as u32,
        pad_d: pad[0],
        pad_h: pad[1],
        pad_w: pad[2],
        pad_t: pad[3],
        dilation_d: dilation[0] as u32,
        dilation_h: dilation[1] as u32,
        dilation_w: dilation[2] as u32,
        dilation_t: dilation[3] as u32,
        out_d: output[0] as u32,
        out_h: output[1] as u32,
        out_w: output[2] as u32,
        out_t: output[3] as u32,
        out_channels: out_channels as u32,
        span: span as u32,
        dims,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.conv.grad_input.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.conv.grad_input.encoder"),
    });

    let pipeline = ctx.fused_grad_input_pipeline()?;
    let bind_group =
        ctx.fused_grad_input_bind_group(&grad_matrix_buf, &weights_buf, &output_buf, &params_buf);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.conv.grad_input.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        let groups_x = (input[2] as u32).div_ceil(GRAD_INPUT_TILE_X);
        let groups_y = (input[1] as u32).div_ceil(GRAD_INPUT_TILE_Y);
        let groups_z = (total_bc as u32).div_ceil(GRAD_INPUT_TILE_Z);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), groups_z.max(1));
    }

    queue.submit(Some(encoder.finish()));

    readback_f32(device, queue, &output_buf, input_volume)
}

#[allow(clippy::too_many_arguments)]
pub fn conv_grad_input_fused(
    grad_matrix: &[f32],
    weights: &[f32],
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
    out_h: usize,
    out_w: usize,
    out_channels: usize,
) -> Result<Vec<f32>, String> {
    conv_grad_input_fused_common(GradInputLaunch {
        grad_matrix,
        weights,
        batch,
        in_channels,
        input: [1, input_h, input_w, 1],
        kernel: [1, kernel_h, kernel_w, 1],
        stride: [1, stride_h, stride_w, 1],
        pad: [0, pad_h, pad_w, 0],
        dilation: [1, dilation_h, dilation_w, 1],
        output: [1, out_h, out_w, 1],
        out_channels,
        dims: 2,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn conv3d_grad_input_fused(
    grad_matrix: &[f32],
    weights: &[f32],
    batch: usize,
    in_channels: usize,
    input_d: usize,
    input_h: usize,
    input_w: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    pad_d: i32,
    pad_h: i32,
    pad_w: i32,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
    out_d: usize,
    out_h: usize,
    out_w: usize,
    out_channels: usize,
) -> Result<Vec<f32>, String> {
    conv_grad_input_fused_common(GradInputLaunch {
        grad_matrix,
        weights,
        batch,
        in_channels,
        input: [input_d, input_h, input_w, 1],
        kernel: [kernel_d, kernel_h, kernel_w, 1],
        stride: [stride_d, stride_h, stride_w, 1],
        pad: [pad_d, pad_h, pad_w, 0],
        dilation: [dilation_d, dilation_h, dilation_w, 1],
        output: [out_d, out_h, out_w, 1],
        out_channels,
        dims: 3,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn conv4d_grad_input_fused(
    grad_matrix: &[f32],
    weights: &[f32],
    batch: usize,
    in_channels: usize,
    input_d: usize,
    input_h: usize,
    input_w: usize,
    input_t: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    kernel_t: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    stride_t: usize,
    pad_d: i32,
    pad_h: i32,
    pad_w: i32,
    pad_t: i32,
    dilation_d: usize,
    dilation_h: usize,
    dilation_w: usize,
    dilation_t: usize,
    out_d: usize,
    out_h: usize,
    out_w: usize,
    out_t: usize,
    out_channels: usize,
) -> Result<Vec<f32>, String> {
    conv_grad_input_fused_common(GradInputLaunch {
        grad_matrix,
        weights,
        batch,
        in_channels,
        input: [input_d, input_h, input_w, input_t],
        kernel: [kernel_d, kernel_h, kernel_w, kernel_t],
        stride: [stride_d, stride_h, stride_w, stride_t],
        pad: [pad_d, pad_h, pad_w, pad_t],
        dilation: [dilation_d, dilation_h, dilation_w, dilation_t],
        output: [out_d, out_h, out_w, out_t],
        out_channels,
        dims: 4,
    })
}

fn lstm_scan_storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn lstm_scan_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

pub fn lstm_backward_scan_shape_supported(timesteps: usize, hidden_dim: usize) -> bool {
    if timesteps == 0 || hidden_dim == 0 {
        return false;
    }
    if timesteps > u32::MAX as usize || hidden_dim > u32::MAX as usize {
        return false;
    }
    hidden_dim
        .checked_mul(4)
        .filter(|gate_width| *gate_width <= u32::MAX as usize)
        .is_some()
}

pub fn lstm_backward_scan_workgroup_size() -> usize {
    LSTM_BACKWARD_SCAN_WORKGROUP as usize
}

pub fn supports_lstm_backward_scan(timesteps: usize, hidden_dim: usize) -> bool {
    lstm_backward_scan_shape_supported(timesteps, hidden_dim) && dense_context().is_ok()
}

#[allow(clippy::too_many_arguments)]
pub fn lstm_backward_scan(
    gates_i: &[f32],
    gates_f: &[f32],
    gates_g: &[f32],
    gates_o: &[f32],
    cell_states: &[f32],
    grad_output: &[f32],
    weight_hh_t: &[f32],
    timesteps: usize,
    hidden_dim: usize,
) -> Result<Vec<f32>, String> {
    if !lstm_backward_scan_shape_supported(timesteps, hidden_dim) {
        return Err("lstm_backward_scan shape unsupported by WGPU helper".into());
    }
    if timesteps == 0 || hidden_dim == 0 {
        return Err("lstm_backward_scan dimensions must be positive".into());
    }
    let gate_width = hidden_dim
        .checked_mul(4)
        .ok_or_else(|| "lstm_backward_scan gate width overflow".to_string())?;
    let hidden_values = timesteps
        .checked_mul(hidden_dim)
        .ok_or_else(|| "lstm_backward_scan hidden volume overflow".to_string())?;
    let gate_values = timesteps
        .checked_mul(gate_width)
        .ok_or_else(|| "lstm_backward_scan gate volume overflow".to_string())?;
    let cell_values = timesteps
        .checked_add(1)
        .and_then(|value| value.checked_mul(hidden_dim))
        .ok_or_else(|| "lstm_backward_scan cell volume overflow".to_string())?;
    let recurrent_values = gate_width
        .checked_mul(hidden_dim)
        .ok_or_else(|| "lstm_backward_scan recurrent weight volume overflow".to_string())?;
    if timesteps > u32::MAX as usize
        || hidden_dim > u32::MAX as usize
        || gate_width > u32::MAX as usize
    {
        return Err("lstm_backward_scan dimensions exceed u32 dispatch range".into());
    }
    for (label, actual, expected) in [
        ("gates_i", gates_i.len(), hidden_values),
        ("gates_f", gates_f.len(), hidden_values),
        ("gates_g", gates_g.len(), hidden_values),
        ("gates_o", gates_o.len(), hidden_values),
        ("cell_states", cell_states.len(), cell_values),
        ("grad_output", grad_output.len(), hidden_values),
        ("weight_hh_t", weight_hh_t.len(), recurrent_values),
    ] {
        if actual != expected {
            return Err(format!(
                "lstm_backward_scan {label} length mismatch: expected {expected}, got {actual}"
            ));
        }
    }

    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();
    let gates_i_buf = upload_lhs(device, "st.tensor.wgpu_dense.lstm_scan.gates_i", gates_i);
    let gates_f_buf = upload_lhs(device, "st.tensor.wgpu_dense.lstm_scan.gates_f", gates_f);
    let gates_g_buf = upload_lhs(device, "st.tensor.wgpu_dense.lstm_scan.gates_g", gates_g);
    let gates_o_buf = upload_lhs(device, "st.tensor.wgpu_dense.lstm_scan.gates_o", gates_o);
    let cell_states_buf = upload_lhs(
        device,
        "st.tensor.wgpu_dense.lstm_scan.cell_states",
        cell_states,
    );
    let grad_output_buf = upload_lhs(
        device,
        "st.tensor.wgpu_dense.lstm_scan.grad_output",
        grad_output,
    );
    let weight_hh_t_buf = upload_lhs(
        device,
        "st.tensor.wgpu_dense.lstm_scan.weight_hh_t",
        weight_hh_t,
    );
    let gate_gradients_buf = allocate_output(
        device,
        "st.tensor.wgpu_dense.lstm_scan.gate_gradients",
        gate_values,
    );
    let scratch_values = hidden_dim
        .checked_mul(2)
        .ok_or_else(|| "lstm_backward_scan scratch volume overflow".to_string())?;
    let scratch = vec![0.0f32; scratch_values];
    let scratch_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.lstm_scan.scratch"),
        contents: bytemuck::cast_slice(&scratch),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let params = LstmBackwardScanParams {
        timesteps: timesteps as u32,
        hidden_dim: hidden_dim as u32,
        gate_width: gate_width as u32,
        _pad0: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.lstm_scan.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("st.tensor.wgpu_dense.lstm_scan.layout"),
        entries: &[
            lstm_scan_storage_entry(0, true),
            lstm_scan_storage_entry(1, true),
            lstm_scan_storage_entry(2, true),
            lstm_scan_storage_entry(3, true),
            lstm_scan_storage_entry(4, true),
            lstm_scan_storage_entry(5, true),
            lstm_scan_storage_entry(6, true),
            lstm_scan_storage_entry(7, false),
            lstm_scan_storage_entry(8, false),
            lstm_scan_uniform_entry(9),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("st.tensor.wgpu_dense.lstm_scan.pipeline_layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let shader = create_wgsl_module(
        device,
        "st.tensor.wgpu_dense.lstm_scan.shader",
        LSTM_BACKWARD_SCAN_WGSL,
    )
    .map_err(|err| err.to_string())?;
    let pipeline = create_compute_pipeline(
        device,
        "st.tensor.wgpu_dense.lstm_scan.pipeline",
        Some(&pipeline_layout),
        &shader,
        "lstm_backward_scan",
    )
    .map_err(|err| err.to_string())?;
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.tensor.wgpu_dense.lstm_scan.bind_group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: gates_i_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gates_f_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gates_g_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: gates_o_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: cell_states_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: grad_output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: weight_hh_t_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: gate_gradients_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: scratch_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.lstm_scan.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.lstm_scan.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit(Some(encoder.finish()));

    let gate_gradients = readback_f32(device, queue, &gate_gradients_buf, gate_values)?;
    if let Some((idx, value)) = gate_gradients
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "lstm_backward_scan produced non-finite gate gradient at {idx}: {value}"
        ));
    }
    let scratch = readback_f32(device, queue, &scratch_buf, scratch_values)?;
    if let Some((idx, value)) = scratch
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "lstm_backward_scan produced non-finite recurrent scratch at {idx}: {value}"
        ));
    }
    Ok(gate_gradients)
}

pub fn ramanujan_pi_gpu(iterations: usize) -> Result<f64, String> {
    let iterations = iterations.max(1);
    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();

    let output_buf = allocate_output(device, "st.tensor.wgpu_dense.ramanujan_pi.output", 1);
    let params = RamanujanPiParams {
        iterations: iterations as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.wgpu_dense.ramanujan_pi.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.wgpu_dense.ramanujan_pi.encoder"),
    });

    let pipeline = ctx.ramanujan_pi_pipeline()?;
    let bind_group = ctx.ramanujan_pi_bind_group(&output_buf, &params_buf);

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.wgpu_dense.ramanujan_pi.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline.as_ref());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(Some(encoder.finish()));

    let values = readback_f32(device, queue, &output_buf, 1)?;
    Ok(values.first().copied().unwrap_or(0.0) as f64)
}

fn fallback_tile_config(rows: usize, inner: usize, cols: usize) -> TileConfig {
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

const MATMUL_AUTOTUNE_REVISION: u64 = 1;
const AUTOTUNE_SAMPLE_MAX_DIM: usize = 1024;
const AUTOTUNE_MIN_VOLUME: usize = 32 * 32 * 32;
const AUTOTUNE_WARMUP_RUNS: usize = 1;
const AUTOTUNE_SAMPLE_RUNS: usize = 3;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct StoredTileConfig {
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
}

impl From<TileConfig> for StoredTileConfig {
    fn from(tile: TileConfig) -> Self {
        Self {
            tile_m: tile.tile_m(),
            tile_n: tile.tile_n(),
            tile_k: tile.tile_k(),
        }
    }
}

impl From<StoredTileConfig> for TileConfig {
    fn from(stored: StoredTileConfig) -> Self {
        TileConfig::new(stored.tile_m, stored.tile_n, stored.tile_k)
    }
}

fn autotune_tile_config(
    ctx: &GpuContext,
    rows: usize,
    inner: usize,
    cols: usize,
) -> Option<TileConfig> {
    if !should_autotune(rows, inner, cols) {
        return None;
    }

    let autotune_enabled = autotune_env_enabled();
    if !autotune_enabled {
        return None;
    }

    let (bucket_rows, bucket_inner, bucket_cols) = quantized_problem(rows, inner, cols);
    let (key, path) = matmul_autotune_key(ctx, bucket_rows, bucket_inner, bucket_cols)?;

    if let Some(tile) = ctx
        .autotune_cache
        .lock()
        .ok()
        .and_then(|guard| guard.get(&key).copied())
    {
        return Some(tile);
    }

    let sample_rows = sample_dimension(bucket_rows);
    let sample_inner = sample_dimension(bucket_inner);
    let sample_cols = sample_dimension(bucket_cols);

    let context = MatmulAutotuneContext {
        rows: bucket_rows,
        inner: bucket_inner,
        cols: bucket_cols,
        sample_rows: sample_rows.min(AUTOTUNE_SAMPLE_MAX_DIM),
        sample_inner: sample_inner.min(AUTOTUNE_SAMPLE_MAX_DIM),
        sample_cols: sample_cols.min(AUTOTUNE_SAMPLE_MAX_DIM),
        revision: MATMUL_AUTOTUNE_REVISION,
        runs: AUTOTUNE_SAMPLE_RUNS as u32,
    };

    eprintln!("[autotune] key={key} apply={autotune_enabled}");

    let matches = if autotune_enabled {
        lookup_similar(path.as_path(), &key, &context, 4)
    } else {
        Vec::new()
    };

    if autotune_enabled {
        let stored = load_best_typed(path.as_path(), &key, &context, None::<StoredTileConfig>);
        if let Some(stored) = stored {
            let tile: TileConfig = stored.into();
            if tile_supported(ctx.device(), tile) {
                if let Ok(mut cache) = ctx.autotune_cache.lock() {
                    cache.insert(key.clone(), tile);
                }
                return Some(tile);
            }
        }
    }

    let device = ctx.device();

    let lhs_len = sample_rows.checked_mul(sample_inner)?;
    let rhs_len = sample_inner.checked_mul(sample_cols)?;
    let out_len = sample_rows.checked_mul(sample_cols)?;

    let lhs_data = vec![0.0f32; lhs_len];
    let rhs_data = vec![0.0f32; rhs_len];
    let lhs_buf = upload_lhs(device, "st.tensor.wgpu_dense.autotune.lhs", &lhs_data);
    let rhs_buffers = upload_weights(device, &rhs_data, sample_inner, sample_cols);
    let out_buf = allocate_output(device, "st.tensor.wgpu_dense.autotune.out", out_len);

    let mut best: Option<(TileConfig, f64)> = None;
    let seeds: Vec<(TileConfig, f64)> = matches
        .iter()
        .filter_map(|m| {
            serde_json::from_value::<StoredTileConfig>(m.entry.params.clone())
                .ok()
                .map(|stored| (TileConfig::from(stored), m.entry.score))
        })
        .collect();
    let mut candidates = candidate_tiles(sample_rows, sample_inner, sample_cols, &seeds);

    for candidate in candidates.drain(..) {
        if !tile_supported(device, candidate) {
            continue;
        }
        match microbenchmark_tile(
            ctx,
            sample_rows,
            sample_inner,
            sample_cols,
            candidate,
            &lhs_buf,
            &rhs_buffers,
            &out_buf,
        ) {
            Ok(score) => {
                let update = best
                    .map(|(_, best_score)| score < best_score)
                    .unwrap_or(true);
                if update {
                    best = Some((candidate, score));
                }
            }
            Err(_) => continue,
        }
    }

    if let Some((tile, score)) = best {
        if let Ok(mut cache) = ctx.autotune_cache.lock() {
            cache.insert(key.clone(), tile);
        }
        if autotune_enabled {
            let stored: StoredTileConfig = tile.into();
            let _ = record_best(path.as_path(), &key, &context, score, &stored);
        }
        Some(tile)
    } else {
        let tile = fallback_tile_config(bucket_rows, bucket_inner, bucket_cols);
        if tile_supported(ctx.device(), tile) {
            if let Ok(mut cache) = ctx.autotune_cache.lock() {
                cache.insert(key.clone(), tile);
            }
            Some(tile)
        } else {
            None
        }
    }
}

fn should_autotune(rows: usize, inner: usize, cols: usize) -> bool {
    if rows == 0 || inner == 0 || cols == 0 {
        return false;
    }
    rows.checked_mul(inner)
        .and_then(|volume| volume.checked_mul(cols))
        .map(|volume| volume >= AUTOTUNE_MIN_VOLUME)
        .unwrap_or(false)
}

fn quantized_problem(rows: usize, inner: usize, cols: usize) -> (usize, usize, usize) {
    (
        quantize_dimension(rows),
        quantize_dimension(inner),
        quantize_dimension(cols),
    )
}

fn quantize_dimension(value: usize) -> usize {
    if value == 0 {
        return 0;
    }
    let step = if value <= 64 {
        8
    } else if value <= 256 {
        16
    } else if value <= 1024 {
        32
    } else {
        64
    };
    ((value + step / 2) / step).max(1) * step
}

fn sample_dimension(value: usize) -> usize {
    quantize_dimension(value).clamp(1, AUTOTUNE_SAMPLE_MAX_DIM)
}

fn tile_supported(device: &Device, tile: TileConfig) -> bool {
    let limits = device.limits();
    let wg_x = tile.tile_n();
    let wg_y = tile.tile_m();
    let invocations = wg_x.saturating_mul(wg_y);
    wg_x <= limits.max_compute_workgroup_size_x
        && wg_y <= limits.max_compute_workgroup_size_y
        && invocations <= limits.max_compute_invocations_per_workgroup
}

fn candidate_tiles(
    rows: usize,
    _inner: usize,
    cols: usize,
    seeds: &[(TileConfig, f64)],
) -> Vec<TileConfig> {
    const BASE: [TileConfig; 12] = [
        TileConfig::new(8, 8, 8),
        TileConfig::new(8, 12, 16),
        TileConfig::new(12, 8, 16),
        TileConfig::new(16, 16, 8),
        TileConfig::new(16, 16, 16),
        TileConfig::new(32, 8, 8),
        TileConfig::new(8, 32, 8),
        TileConfig::new(24, 12, 16),
        TileConfig::new(12, 24, 16),
        TileConfig::new(32, 8, 16),
        TileConfig::new(8, 32, 16),
        TileConfig::new(16, 24, 16),
    ];

    let mut ordered = Vec::with_capacity(BASE.len());
    if rows > cols.saturating_mul(2) {
        ordered.push(TileConfig::new(32, 8, 16));
        ordered.push(TileConfig::new(24, 12, 16));
    } else if cols > rows.saturating_mul(2) {
        ordered.push(TileConfig::new(8, 32, 16));
        ordered.push(TileConfig::new(12, 24, 16));
    }

    for (seed_cfg, _) in seeds {
        if !ordered.contains(seed_cfg) {
            ordered.insert(0, *seed_cfg);
        }
    }

    for candidate in BASE {
        if !ordered.contains(&candidate) {
            ordered.push(candidate);
        }
    }

    let mut scored: Vec<(f64, TileConfig)> = ordered
        .into_iter()
        .map(|candidate| (score_candidate(candidate, seeds), candidate))
        .collect();

    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    scored.into_iter().map(|(_, cfg)| cfg).collect()
}

fn score_candidate(candidate: TileConfig, seeds: &[(TileConfig, f64)]) -> f64 {
    if seeds.is_empty() {
        return 0.0;
    }
    seeds
        .iter()
        .map(|(seed_cfg, score)| score + tile_distance(candidate, *seed_cfg))
        .fold(f64::INFINITY, f64::min)
}

fn tile_distance(a: TileConfig, b: TileConfig) -> f64 {
    let mut delta = 0.0;
    delta += rel_diff(a.tile_m(), b.tile_m());
    delta += rel_diff(a.tile_n(), b.tile_n());
    delta += rel_diff(a.tile_k(), b.tile_k());
    delta
}

fn rel_diff(a: u32, b: u32) -> f64 {
    let lhs = a as f64;
    let rhs = b as f64;
    let base = lhs.max(rhs).max(1.0);
    (lhs - rhs).abs() / base
}

#[derive(Serialize)]
struct MatmulAutotuneContext {
    rows: usize,
    inner: usize,
    cols: usize,
    sample_rows: usize,
    sample_inner: usize,
    sample_cols: usize,
    revision: u64,
    runs: u32,
}

#[allow(clippy::too_many_arguments)]
fn microbenchmark_tile(
    ctx: &GpuContext,
    rows: usize,
    inner: usize,
    cols: usize,
    tile: TileConfig,
    lhs: &Buffer,
    rhs: &WeightBuffers,
    out: &Buffer,
) -> Result<f64, String> {
    let (rhs_buffer, rhs_dtype, rhs_scales) = rhs.as_binding();

    for _ in 0..AUTOTUNE_WARMUP_RUNS {
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("st.tensor.wgpu_dense.autotune.warmup"),
            });
        dispatch_matmul(
            ctx,
            &mut encoder,
            lhs,
            rhs_buffer,
            rhs_dtype,
            rhs_scales,
            out,
            rows,
            inner,
            cols,
            tile,
            false,
            0,
            None,
            None,
            false,
            1.0,
        )?;
        ctx.queue().submit(Some(encoder.finish()));
        ctx.device().poll(wgpu::Maintain::Wait);
    }

    let mut total = Duration::default();
    for _ in 0..AUTOTUNE_SAMPLE_RUNS {
        let mut encoder = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("st.tensor.wgpu_dense.autotune.sample"),
            });
        dispatch_matmul(
            ctx,
            &mut encoder,
            lhs,
            rhs_buffer,
            rhs_dtype,
            rhs_scales,
            out,
            rows,
            inner,
            cols,
            tile,
            false,
            0,
            None,
            None,
            false,
            1.0,
        )?;
        let command = encoder.finish();
        let start = Instant::now();
        ctx.queue().submit(Some(command));
        ctx.device().poll(wgpu::Maintain::Wait);
        total += start.elapsed();
    }

    if AUTOTUNE_SAMPLE_RUNS == 0 {
        return Ok(0.0);
    }

    Ok(total.as_secs_f64() / AUTOTUNE_SAMPLE_RUNS as f64)
}

fn encode_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' => ch,
            ' ' => '-',
            _ => '_',
        })
        .collect()
}

fn matmul_autotune_key(
    ctx: &GpuContext,
    rows: usize,
    inner: usize,
    cols: usize,
) -> Option<(String, PathBuf)> {
    let path = autotune_store_path()?;
    let info = ctx.adapter_info();
    let backend = encode_component(&format!("{:?}", info.backend));
    let driver = encode_component(&info.driver);
    let driver_info = encode_component(&info.driver_info);
    let name = encode_component(&info.name);
    let key = format!(
        "wgpu.matmul.v{MATMUL_AUTOTUNE_REVISION:02}|{name}|{vendor:04x}|{device:04x}|{backend}|{driver}|{driver_info}|{rows}x{inner}x{cols}|runs{AUTOTUNE_SAMPLE_RUNS}",
        vendor = info.vendor,
        device = info.device,
    );
    Some((key, path))
}

fn autotune_env_enabled() -> bool {
    env::var("SPIRALTORCH_AUTOTUNE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

fn autotune_store_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os("SPIRALTORCH_AUTOTUNE_STORE") {
        return Some(PathBuf::from(path));
    }
    if let Some(home) = env::var_os("HOME") {
        let mut path = PathBuf::from(home);
        path.push(".spiraltorch");
        path.push("kernels.json");
        return Some(path);
    }
    None
}

pub fn should_use(rows: usize, inner: usize, cols: usize) -> bool {
    if rows == 0 || inner == 0 || cols == 0 {
        return false;
    }

    let volume = rows.saturating_mul(inner).saturating_mul(cols);

    volume >= 32 * 32 * 32
}
