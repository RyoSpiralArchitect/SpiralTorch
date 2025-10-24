//! Lightweight wrappers around WGPU compute pipelines used by SpiralTorch.
//! The module exposes helper routines to load WGSL shaders from disk and
//! construct the compute pipelines that power higher level tensor operators.

pub mod attention;
pub mod compaction;
pub mod compaction2ce;
pub mod compaction_2ce;
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
