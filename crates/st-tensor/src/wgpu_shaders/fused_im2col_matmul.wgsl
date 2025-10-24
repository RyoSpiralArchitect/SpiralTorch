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

override TILE_SIZE : u32 = {tile_size}u;

var<workgroup> lhs_tile : array<f32, TILE_SIZE * TILE_SIZE>;
var<workgroup> rhs_tile : array<f32, TILE_SIZE * TILE_SIZE>;

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

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
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

    let spatial = params.out_h * params.out_w;
    let batch_index = row / spatial;
    let spatial_index = row - batch_index * spatial;
    let out_y = spatial_index / params.out_w;
    let out_x = spatial_index % params.out_w;

    let local_row = lid.y;
    let local_col = lid.x;

    var acc : f32 = 0.0;
    let tiles = (params.span + TILE_SIZE - 1u) / TILE_SIZE;
    var tile_index : u32 = 0u;
    loop {
        if (tile_index >= tiles) {
            break;
        }

        let k_base = tile_index * TILE_SIZE;

        let lhs_k = k_base + local_col;
        lhs_tile[local_row * TILE_SIZE + local_col] =
            load_patch_value(batch_index, out_y, out_x, lhs_k);

        let rhs_k = k_base + local_row;
        var rhs_value : f32 = 0.0;
        if (rhs_k < params.span) {
            rhs_value = weights[col * params.span + rhs_k];
        }
        rhs_tile[local_row * TILE_SIZE + local_col] = rhs_value;

        workgroupBarrier();

        var k : u32 = 0u;
        loop {
            if (k >= TILE_SIZE) {
                break;
            }
            let lhs_val = lhs_tile[local_row * TILE_SIZE + k];
            let rhs_val = rhs_tile[k * TILE_SIZE + local_col];
            acc = acc + lhs_val * rhs_val;
            k = k + 1u;
        }

        workgroupBarrier();
        tile_index = tile_index + 1u;
    }

    let out_index = row * params.out_channels + col;
    output_tensor[out_index] = acc;
}
