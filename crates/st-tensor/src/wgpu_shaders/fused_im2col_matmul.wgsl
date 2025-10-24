// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

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
};

@group(0) @binding(0) var<storage, read> input_tensor : array<f32>;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> output_tensor : array<f32>;
@group(0) @binding(3) var<uniform> params : ConvGemmParams;

override TILE_M : u32 = {tile_m}u;
override TILE_N : u32 = {tile_n}u;
override TILE_K : u32 = {tile_k}u;

var<workgroup> lhs_tile : array<f32, TILE_M * TILE_K>;
var<workgroup> rhs_tile : array<f32, TILE_K * TILE_N>;

fn load_patch_value(
    batch_index: u32,
    out_y: u32,
    out_x: u32,
    k: u32,
) -> f32 {
    if (k >= params.span) {
        return 0.0;
    }
    let kernel_hw = params.kernel_h * params.kernel_w;
    let channel = k / kernel_hw;
    let rem = k % kernel_hw;
    let kernel_y = rem / params.kernel_w;
    let kernel_x = rem % params.kernel_w;

    let base_y = i32(out_y * params.stride_h) - params.pad_h;
    let base_x = i32(out_x * params.stride_w) - params.pad_w;
    let in_y = base_y + i32(kernel_y * params.dilation_h);
    let in_x = base_x + i32(kernel_x * params.dilation_w);

    if (in_y < 0 || in_y >= i32(params.input_h)) {
        return 0.0;
    }
    if (in_x < 0 || in_x >= i32(params.input_w)) {
        return 0.0;
    }

    let in_y_u = u32(in_y);
    let in_x_u = u32(in_x);
    let index = (((batch_index * params.in_channels + channel) * params.input_h + in_y_u)
        * params.input_w)
        + in_x_u;
    return input_tensor[index];
}

fn decode_row(index: u32) -> vec3<u32> {
    let spatial = params.out_h * params.out_w;
    let batch_index = index / spatial;
    let rem = index - batch_index * spatial;
    let out_y = rem / params.out_w;
    let out_x = rem % params.out_w;
    return vec3<u32>(batch_index, out_y, out_x);
}

@compute @workgroup_size(TILE_N, TILE_M, 1)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let total_rows = params.batch * params.out_h * params.out_w;
    if (row >= total_rows || col >= params.out_channels) {
        return;
    }

    let local_m = lid.y;
    let local_n = lid.x;
    let local_linear = local_m * TILE_N + local_n;
    let threads = TILE_M * TILE_N;
    let tile_row_origin = gid.y - local_m;
    let tile_col_origin = gid.x - local_n;

    var acc : f32 = 0.0;
    let tiles = (params.span + TILE_K - 1u) / TILE_K;
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
            if (global_row < total_rows && global_k < params.span) {
                let coords = decode_row(global_row);
                lhs_value = load_patch_value(coords.x, coords.y, coords.z, global_k);
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
            if (global_k < params.span && global_col < params.out_channels) {
                rhs_value = weights[global_col * params.span + global_k];
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
            let lhs_val = lhs_tile[local_m * TILE_K + k];
            let rhs_val = rhs_tile[k * TILE_N + local_n];
            acc = acc + lhs_val * rhs_val;
            k = k + 1u;
        }

        workgroupBarrier();
        tile_index = tile_index + 1u;
    }

    let out_index = row * params.out_channels + col;
    output_tensor[out_index] = acc;
}
