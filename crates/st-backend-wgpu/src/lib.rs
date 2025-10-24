//! Lightweight wrappers around WGPU compute pipelines used by SpiralTorch.
//! The module exposes helper routines to load WGSL shaders from disk and
//! construct the compute pipelines that power higher level tensor operators.

pub mod attention;
pub mod compaction;
pub mod compaction2ce;
pub mod compaction_2ce;
pub mod gelu_back;
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
    create_pipelines as create_softmax_pipelines, upload_params as upload_softmax_params,
    Builder as SoftmaxBuilder, Dispatch as SoftmaxDispatch, Params as SoftmaxParams,
    Pipelines as SoftmaxPipelines,
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
