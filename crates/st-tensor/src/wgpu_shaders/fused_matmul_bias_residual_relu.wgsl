// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct MatmulParams {
    rows: u32,
    cols: u32,
    inner: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> lhs : array<f32>;
@group(0) @binding(1) var<storage, read> rhs : array<f32>;
@group(0) @binding(2) var<storage, read> bias : array<f32>;
@group(0) @binding(3) var<storage, read> residual : array<f32>;
@group(0) @binding(4) var<storage, read_write> out : array<f32>;
@group(0) @binding(5) var<uniform> params : MatmulParams;

override TILE_M : u32 = {tile_m}u;
override TILE_N : u32 = {tile_n}u;
override TILE_K : u32 = {tile_k}u;

var<workgroup> lhs_tile : array<f32, TILE_M * TILE_K>;
var<workgroup> rhs_tile_T : array<f32, TILE_N * TILE_K>;
var<workgroup> bias_tile : array<f32, TILE_N>;

@compute @workgroup_size(TILE_N, TILE_M, 1)
fn main(
    @builtin(workgroup_id) wid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let local_m = lid.y;
    let local_n = lid.x;
    let tile_row_origin = wid.y * TILE_M;
    let tile_col_origin = wid.x * TILE_N;
    let row = tile_row_origin + local_m;
    let col = tile_col_origin + local_n;
    if (row >= params.rows || col >= params.cols) {
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

        var tm : u32 = local_m;
        loop {
            if (tm >= TILE_M) {
                break;
            }
            var tk : u32 = local_n;
            loop {
                if (tk >= TILE_K) {
                    break;
                }
                let global_row = tile_row_origin + tm;
                let global_k = k_base + tk;
                var lhs_value : f32 = 0.0;
                if (global_row < params.rows && global_k < params.inner) {
                    lhs_value = lhs[global_row * params.inner + global_k];
                }
                lhs_tile[tm * TILE_K + tk] = lhs_value;
                tk = tk + TILE_N;
            }
            tm = tm + TILE_M;
        }

        var tn : u32 = local_n;
        loop {
            if (tn >= TILE_N) {
                break;
            }
            var tk : u32 = local_m;
            loop {
                if (tk >= TILE_K) {
                    break;
                }
                let global_k = k_base + tk;
                let global_col = tile_col_origin + tn;
                var rhs_value : f32 = 0.0;
                if (global_k < params.inner && global_col < params.cols) {
                    rhs_value = rhs[global_k * params.cols + global_col];
                }
                rhs_tile_T[tn * TILE_K + tk] = rhs_value;
                tk = tk + TILE_M;
            }
            tn = tn + TILE_N;
        }

        workgroupBarrier();

        let remaining = params.inner - min(params.inner, k_base);
        let k_limit = min(TILE_K, remaining);
        var k : u32 = 0u;
        loop {
            if (k >= k_limit) {
                break;
            }
            let lhs_tile_value = lhs_tile[local_m * TILE_K + k];
            let rhs_tile_value = rhs_tile_T[local_n * TILE_K + k];
            acc = acc + lhs_tile_value * rhs_tile_value;
            k = k + 1u;
        }

        workgroupBarrier();
        tile_index = tile_index + 1u;
    }

    if (local_m == 0u) {
        var bias_value : f32 = 0.0;
        if (col < params.cols) {
            bias_value = bias[col];
        }
        bias_tile[local_n] = bias_value;
    }
    workgroupBarrier();
    let bias_value = bias_tile[local_n];

    let out_index = row * params.cols + col;
    let residual_value = residual[out_index];
    let sum = acc + bias_value + residual_value;
    let activated = max(sum, 0.0);
    out[out_index] = activated;
}
