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
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<uniform> params : MatmulParams;

override TILE_M : u32 = {tile_m}u;
override TILE_N : u32 = {tile_n}u;
override TILE_K : u32 = {tile_k}u;

var<workgroup> lhs_tile : array<f32, TILE_M * TILE_K>;
var<workgroup> rhs_tile : array<f32, TILE_K * TILE_N>;

@compute @workgroup_size(TILE_N, TILE_M, 1)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.rows || col >= params.cols) {
        return;
    }

    let local_m = lid.y;
    let local_n = lid.x;
    let local_linear = local_m * TILE_N + local_n;
    let threads = TILE_M * TILE_N;
    let tile_row_origin = gid.y - local_m;
    let tile_col_origin = gid.x - local_n;

    var acc : f32 = 0.0;
    let tiles = (params.inner + TILE_K - 1u) / TILE_K;
    var tile_index : u32 = 0u;
    loop {
        if (tile_index >= tiles) {
            break;
        }

        let k_base = tile_index * TILE_K;

        var lhs_iter : u32 = local_linear;
        loop {
            if (lhs_iter >= TILE_M * TILE_K) {
                break;
            }
            let tile_row = lhs_iter / TILE_K;
            let tile_k = lhs_iter % TILE_K;
            let global_row = tile_row_origin + tile_row;
            let global_k = k_base + tile_k;
            var lhs_value : f32 = 0.0;
            if (global_row < params.rows && global_k < params.inner) {
                lhs_value = lhs[global_row * params.inner + global_k];
            }
            lhs_tile[lhs_iter] = lhs_value;
            lhs_iter = lhs_iter + threads;
        }

        var rhs_iter : u32 = local_linear;
        loop {
            if (rhs_iter >= TILE_K * TILE_N) {
                break;
            }
            let tile_k = rhs_iter / TILE_N;
            let tile_col = rhs_iter % TILE_N;
            let global_k = k_base + tile_k;
            let global_col = tile_col_origin + tile_col;
            var rhs_value : f32 = 0.0;
            if (global_k < params.inner && global_col < params.cols) {
                rhs_value = rhs[global_k * params.cols + global_col];
            }
            rhs_tile[rhs_iter] = rhs_value;
            rhs_iter = rhs_iter + threads;
        }

        workgroupBarrier();

        var k : u32 = 0u;
        loop {
            if (k >= TILE_K) {
                break;
            }
            let lhs_tile_value = lhs_tile[local_m * TILE_K + k];
            let rhs_tile_value = rhs_tile[k * TILE_N + local_n];
            acc = acc + lhs_tile_value * rhs_tile_value;
            k = k + 1u;
        }

        workgroupBarrier();
        tile_index = tile_index + 1u;
    }

    let out_index = row * params.cols + col;
    out[out_index] = acc;
}
