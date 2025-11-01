//! Lightweight wrappers around WGPU compute pipelines used by SpiralTorch.
//! The module exposes helper routines to load WGSL shaders from disk and
//! construct the compute pipelines that power higher level tensor operators.

pub mod attention;
pub mod compaction;
pub mod compaction2ce;
pub mod compaction_2ce;
pub mod gelu_back;
pub mod hardmax;
pub mod middlemax;
pub mod midk_bottomk;
pub mod nd_indexer;
pub mod nerf;
pub mod render;
pub mod softmax;
pub mod topk_keepk;
pub mod transform;

mod util;

pub use midk_bottomk::{
    dispatch as dispatch_midk_bottomk, encode as encode_midk_bottomk,
    encode_into as encode_midk_bottomk_into, ApplyStrategy as MidkBottomkApplyStrategy,
    DispatchArgs as MidkBottomkDispatchArgs,
    DispatchValidationError as MidkBottomkDispatchValidationError,
    ElementCounts as MidkBottomkElementCounts, Kind as MidkBottomkKind,
};

pub use middlemax::{
    create_pipelines as create_middlemax_pipelines, dispatch as dispatch_middlemax,
    encode as encode_middlemax, encode_into as encode_middlemax_into,
    ApplyStrategy as MiddlemaxApplyStrategy, Builder as MiddlemaxBuilder,
    DispatchArgs as MiddlemaxDispatchArgs,
    DispatchValidationError as MiddlemaxDispatchValidationError,
    ElementCounts as MiddlemaxElementCounts, Kind as MiddlemaxKind,
    Pipelines as MiddlemaxPipelines,
};

pub use hardmax::{
    create_pipelines as create_hardmax_pipelines, dispatch as dispatch_hardmax,
    encode as encode_hardmax, encode_into as encode_hardmax_into,
    upload_params as upload_hardmax_params, Builder as HardmaxBuilder, Dispatch as HardmaxDispatch,
    DispatchArgs as HardmaxDispatchArgs, MissingMaskBufferError as HardmaxMissingMaskBufferError,
    Mode as HardmaxMode, Params as HardmaxParams, PipelineVariant as HardmaxPipelineVariant,
    Pipelines as HardmaxPipelines,
};

pub use util::{
    load_compute_pipeline, load_compute_pipeline_with_layout, read_wgsl, ShaderCache,
    ShaderLoadError,
};

pub use attention::{
    fused_attention, AccumulatorPrecision as FusedAttentionAccumulatorPrecision,
    Kernel as FusedAttentionKernel, Params as FusedAttentionParams, Plan as FusedAttentionPlan,
    PlanError as FusedAttentionPlanError, FLAG_USE_ATTN_BIAS, FLAG_USE_Z_BIAS,
};

pub use softmax::{
    create_pipelines as create_softmax_pipelines, dispatch as dispatch_softmax,
    encode as encode_softmax, encode_into as encode_softmax_into,
    upload_params as upload_softmax_params, Builder as SoftmaxBuilder, Dispatch as SoftmaxDispatch,
    DispatchArgs as SoftmaxDispatchArgs, Params as SoftmaxParams,
    PipelineVariant as SoftmaxPipelineVariant, Pipelines as SoftmaxPipelines,
};

pub use gelu_back::{
    create_pipelines as create_gelu_back_pipelines,
    create_pipelines_with_geometry as create_gelu_back_pipelines_with_geometry,
    upload_fused_uniforms as upload_gelu_back_fused_uniforms,
    upload_reduce_uniforms as upload_gelu_back_reduce_uniforms, Builder as GeluBackBuilder,
    FusedUniforms as GeluBackFusedUniforms, Geometry as GeluBackGeometry,
    Pipelines as GeluBackPipelines, ReduceUniforms as GeluBackReduceUniforms,
    DEFAULT_REDUCE_WG as GELU_BACK_DEFAULT_REDUCE_WG, DEFAULT_WG_COLS as GELU_BACK_DEFAULT_WG_COLS,
    DEFAULT_WG_ROWS as GELU_BACK_DEFAULT_WG_ROWS,
};
