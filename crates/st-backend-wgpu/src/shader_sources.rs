//! Canonical WGSL sources owned by the WGPU backend.
//!
//! Higher layers may specialise these templates, but should not reach into the
//! backend crate's source tree with relative `include_str!` paths.

pub const SOFTMAX_WORKGROUP_WGSL: &str = include_str!("shaders/softmax_workgroup.wgsl");
pub const SOFTMAX_SUBGROUP_WGSL: &str = include_str!("shaders/softmax_subgroup.wgsl");
pub const SOFTMAX_ZSPACE_PROJECTION_WGSL: &str =
    include_str!("shaders/softmax_zspace_projection.wgsl");
pub const SOFTMAX_SPIRAL_CONSENSUS_WGSL: &str =
    include_str!("shaders/softmax_spiral_consensus.wgsl");
pub const FUSED_ATTENTION_ONLINE_WGSL: &str = include_str!("shaders/fused_attention_online.wgsl");
