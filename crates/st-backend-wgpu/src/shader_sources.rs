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

pub const DENSE_MATMUL_WGSL: &str = include_str!("shaders/dense_matmul.wgsl");

/// Low-level specializer; callers validate tile and dispatch limits before use.
/// Float32 is shared across host-tensor and resident workloads. Int8 remains an
/// explicit option for the existing approximate host-tensor path only.
pub fn dense_matmul_source(tile: [u32; 3], int8: bool) -> String {
    let [m, n, k] = tile;
    let rhs_load = if int8 {
        "let stride = (params.inner + 3u) / 4u;
        let base = col * stride + (k >> 2u);
        let word = rhs_packed[base];
        let lane = (k & 3u) * 8u;
        let byte_val = (word >> lane) & 0xFFu;
        let signed_val = bitcast<i32>(byte_val << 24u) >> 24;
        let scale = scales[col];
        return f32(signed_val) * scale;"
    } else {
        "return rhs_packed[k * params.cols + col];"
    };
    DENSE_MATMUL_WGSL
        .replace("{f16_enable}", "")
        .replace(
            "{rhs_storage_type}",
            if int8 { "array<u32>" } else { "array<f32>" },
        )
        .replace("{rhs_load_body}", rhs_load)
        .replace("{tile_m}", &m.to_string())
        .replace("{tile_n}", &n.to_string())
        .replace("{tile_k}", &k.to_string())
        .replace("{tile_mk}", &format!("{}u", u64::from(m) * u64::from(k)))
        .replace("{tile_nk}", &format!("{}u", u64::from(n) * u64::from(k)))
        .replace("{workgroup_size_x}", &n.to_string())
        .replace("{workgroup_size_y}", &m.to_string())
        .replace("{fma_line}", "acc = acc + lhs_val * rhs_val;")
}
