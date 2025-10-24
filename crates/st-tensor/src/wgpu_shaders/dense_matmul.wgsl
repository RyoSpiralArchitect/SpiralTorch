{f16_enable}
struct MatmulUniforms {
    rows: u32;
    cols: u32;
    inner: u32;
    flags: u32;
};

const FLAG_USE_BIAS: u32 = 1u << 0u;
const FLAG_FUSED_RELU: u32 = 1u << 1u;
const FLAG_FUSED_GELU: u32 = 1u << 2u;
const FLAG_FUSED_RESIDUAL: u32 = 1u << 3u;

@group(0) @binding(0) var<storage, read> lhs : array<f32>;
@group(0) @binding(1) var<storage, read> rhs_packed : {rhs_storage_type};
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<storage, read> bias : array<f32>;
@group(0) @binding(4) var<storage, read> residual : array<f32>;
@group(0) @binding(5) var<storage, read> scales : array<f32>;
@group(0) @binding(6) var<uniform> params : MatmulUniforms;

const TILE_M : u32 = {tile_m}u;
const TILE_N : u32 = {tile_n}u;
const TILE_K : u32 = {tile_k}u;
const WG_SIZE_X : u32 = {workgroup_size_x}u;
const WG_SIZE_Y : u32 = {workgroup_size_y}u;

var<workgroup> tile_a : array<f32, TILE_M * TILE_K>;
var<workgroup> tile_b : array<f32, TILE_N * TILE_K>;

fn load_rhs_value(k : u32, col : u32) -> f32 {{
    {rhs_load_body}
}}

fn apply_fusions(acc: f32, index: u32, col: u32) -> f32 {{
    var value = acc;
    if ((params.flags & FLAG_USE_BIAS) != 0u) {{
        value = value + bias[col];
    }}
    if ((params.flags & FLAG_FUSED_RESIDUAL) != 0u) {{
        value = value + residual[index];
    }}
    if ((params.flags & FLAG_FUSED_GELU) != 0u) {{
        let x = value;
        let coeff = 0.044715;
        let sqrt_two_over_pi = 0.7978845834732056;
        let y = sqrt_two_over_pi * (x + coeff * x * x * x);
        value = 0.5 * x * (1.0 + tanh(y));
    }} else if ((params.flags & FLAG_FUSED_RELU) != 0u) {{
        value = max(value, 0.0);
    }}
    return value;
}}

@compute @workgroup_size({workgroup_size_x}, {workgroup_size_y}, 1)
fn main(
    @builtin(workgroup_id) wid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let tile_row = wid.y * TILE_M;
    let tile_col = wid.x * TILE_N;
    let local_row = lid.y;
    let local_col = lid.x;
    let global_row = tile_row + local_row;
    let global_col = tile_col + local_col;

    if (global_row >= params.rows || global_col >= params.cols) {
        return;
    }

    var acc : f32 = 0.0;
    let tiles = (params.inner + TILE_K - 1u) / TILE_K;
    var tile_index : u32 = 0u;
    loop {
        if (tile_index >= tiles) {
            break;
        }
        let k_base = tile_index * TILE_K;

        var load_row = local_row;
        loop {
            if (load_row >= TILE_M) {
                break;
            }
            var load_k = local_col;
            loop {
                if (load_k >= TILE_K) {
                    break;
                }
                let a_row = tile_row + load_row;
                let a_k = k_base + load_k;
                var value = 0.0;
                if (a_row < params.rows && a_k < params.inner) {
                    value = lhs[a_row * params.inner + a_k];
                }
                tile_a[load_row * TILE_K + load_k] = value;
                load_k = load_k + WG_SIZE_X;
            }
            load_row = load_row + WG_SIZE_Y;
        }

        var load_col = local_col;
        loop {
            if (load_col >= TILE_N) {
                break;
            }
            var load_k = local_row;
            loop {
                if (load_k >= TILE_K) {
                    break;
                }
                let b_k = k_base + load_k;
                let b_col = tile_col + load_col;
                var value = 0.0;
                if (b_k < params.inner && b_col < params.cols) {
                    value = load_rhs_value(b_k, b_col);
                }
                tile_b[load_col * TILE_K + load_k] = value;
                load_k = load_k + WG_SIZE_Y;
            }
            load_col = load_col + WG_SIZE_X;
        }

        workgroupBarrier();

        var k : u32 = 0u;
        loop {
            if (k >= TILE_K || (k_base + k) >= params.inner) {
                break;
            }
            let lhs_val = tile_a[local_row * TILE_K + k];
            let rhs_val = tile_b[local_col * TILE_K + k];
            {fma_line}
            k = k + 1u;
        }

        workgroupBarrier();
        tile_index = tile_index + 1u;
    }

    let index = global_row * params.cols + global_col;
    out[index] = apply_fusions(acc, index, global_col);
}
