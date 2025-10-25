// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct GradInputParams {
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
    out_channels: u32,
    span: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read> grad_matrix : array<f32>;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(3) var<uniform> params : GradInputParams;

const TILE_X : u32 = {tile_x}u;
const TILE_Y : u32 = {tile_y}u;
const TILE_Z : u32 = {tile_z}u;

fn linear_index(batch: u32, channel: u32, y: u32, x: u32) -> u32 {
    return (((batch * params.in_channels + channel) * params.input_h + y)
        * params.input_w)
        + x;
}

fn weight_index(channel: u32, kernel_y: u32, kernel_x: u32) -> u32 {
    let kernel_hw = params.kernel_h * params.kernel_w;
    return channel * kernel_hw + kernel_y * params.kernel_w + kernel_x;
}

@compute @workgroup_size(TILE_X, TILE_Y, TILE_Z)
fn main(
    @builtin(workgroup_id) wid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let in_x = wid.x * TILE_X + lid.x;
    let in_y = wid.y * TILE_Y + lid.y;
    let bc = wid.z * TILE_Z + lid.z;
    if (in_x >= params.input_w || in_y >= params.input_h) {
        return;
    }
    let total_bc = params.batch * params.in_channels;
    if (bc >= total_bc) {
        return;
    }

    let channel = bc % params.in_channels;
    let batch_index = bc / params.in_channels;

    var acc : f32 = 0.0;
    let input_y_i = i32(in_y);
    let input_x_i = i32(in_x);
    let stride_h = i32(params.stride_h);
    let stride_w = i32(params.stride_w);

    var ky : u32 = 0u;
    loop {
        if (ky >= params.kernel_h) {
            break;
        }
        let base_y = input_y_i + params.pad_h - i32(ky * params.dilation_h);
        if (base_y >= 0 && stride_h > 0) {
            let div = base_y / stride_h;
            let rem = base_y - div * stride_h;
            if (rem == 0 && div >= 0 && div < i32(params.out_h)) {
                let out_y = u32(div);
                var kx : u32 = 0u;
                loop {
                    if (kx >= params.kernel_w) {
                        break;
                    }
                    let base_x = input_x_i + params.pad_w - i32(kx * params.dilation_w);
                    if (base_x >= 0 && stride_w > 0) {
                        let div_x = base_x / stride_w;
                        let rem_x = base_x - div_x * stride_w;
                        if (rem_x == 0 && div_x >= 0 && div_x < i32(params.out_w)) {
                            let out_x = u32(div_x);
                            let row = batch_index * params.out_h * params.out_w
                                + out_y * params.out_w
                                + out_x;
                            let grad_row_offset = row * params.out_channels;
                            let weight_offset = weight_index(channel, ky, kx);
                            var oc : u32 = 0u;
                            loop {
                                if (oc >= params.out_channels) {
                                    break;
                                }
                                let grad_value = grad_matrix[grad_row_offset + oc];
                                let w_idx = oc * params.span + weight_offset;
                                let weight_value = weights[w_idx];
                                acc = acc + grad_value * weight_value;
                                oc = oc + 1u;
                            }
                        }
                    }
                    kx = kx + 1u;
                }
            }
        }
        ky = ky + 1u;
    }

    let index = linear_index(batch_index, channel, in_y, in_x);
    grad_input[index] = acc;
}
