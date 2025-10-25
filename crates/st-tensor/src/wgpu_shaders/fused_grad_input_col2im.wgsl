// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct GradInputParams {
    batch: u32,
    in_channels: u32,
    input_d: u32,
    input_h: u32,
    input_w: u32,
    input_t: u32,
    kernel_d: u32,
    kernel_h: u32,
    kernel_w: u32,
    kernel_t: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    stride_t: u32,
    pad_d: i32,
    pad_h: i32,
    pad_w: i32,
    pad_t: i32,
    dilation_d: u32,
    dilation_h: u32,
    dilation_w: u32,
    dilation_t: u32,
    out_d: u32,
    out_h: u32,
    out_w: u32,
    out_t: u32,
    out_channels: u32,
    span: u32,
    dims: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> grad_matrix : array<f32>;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(3) var<uniform> params : GradInputParams;

const TILE_X : u32 = {tile_x}u;
const TILE_Y : u32 = {tile_y}u;
const TILE_Z : u32 = {tile_z}u;
const RAMANUJAN_PI_6 : f32 = {ramanujan_pi_6};

fn linear_index(
    batch: u32,
    channel: u32,
    depth: u32,
    y: u32,
    x: u32,
    time: u32,
) -> u32 {
    let channels = batch * params.in_channels + channel;
    let depth_index = channels * params.input_d + depth;
    let height_index = depth_index * params.input_h + y;
    let width_index = height_index * params.input_w + x;
    return width_index * params.input_t + time;
}

fn weight_index(channel: u32, kd: u32, kh: u32, kw: u32, kt: u32) -> u32 {
    let kernel_hw = params.kernel_h * params.kernel_w;
    let kernel_volume = params.kernel_d * kernel_hw * params.kernel_t;
    let depth_offset = kd * kernel_hw * params.kernel_t;
    let height_offset = kh * params.kernel_w * params.kernel_t;
    let width_offset = kw * params.kernel_t;
    return channel * kernel_volume + depth_offset + height_offset + width_offset + kt;
}

fn output_row(
    batch: u32,
    out_d: u32,
    out_h: u32,
    out_w: u32,
    out_t: u32,
) -> u32 {
    let depth_index = ((batch * params.out_d) + out_d) * params.out_h + out_h;
    let width_index = (depth_index * params.out_w) + out_w;
    return width_index * params.out_t + out_t;
}

@compute @workgroup_size(TILE_X, TILE_Y, TILE_Z)
fn main(
    @builtin(workgroup_id) wid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let in_x = wid.x * TILE_X + lid.x;
    let in_y = wid.y * TILE_Y + lid.y;
    if (in_x >= params.input_w || in_y >= params.input_h) {
        return;
    }
    let total_bc = params.batch * params.in_channels * params.input_d * params.input_t;
    let bc = wid.z * TILE_Z + lid.z;
    if (bc >= total_bc) {
        return;
    }
    let depth_time = params.input_d * params.input_t;
    let depth_time_index = bc % depth_time;
    let depth = depth_time_index / params.input_t;
    let time = depth_time_index % params.input_t;
    let channel_batch = bc / depth_time;
    let channel = channel_batch % params.in_channels;
    let batch_index = channel_batch / params.in_channels;

    var acc : f32 = 0.0;
    let input_d_i = i32(depth);
    let input_y_i = i32(in_y);
    let input_x_i = i32(in_x);
    let input_t_i = i32(time);
    let stride_d = i32(params.stride_d);
    let stride_h = i32(params.stride_h);
    let stride_w = i32(params.stride_w);

    var kd : u32 = 0u;
    loop {
        if (kd >= params.kernel_d) {
            break;
        }
        let base_d = input_d_i + params.pad_d - i32(kd * params.dilation_d);
        if (base_d >= 0 && stride_d > 0) {
            let div_d = base_d / stride_d;
            let rem_d = base_d - div_d * stride_d;
            if (rem_d == 0 && div_d >= 0 && div_d < i32(params.out_d)) {
                let out_d = u32(div_d);
                var ky : u32 = 0u;
                loop {
                    if (ky >= params.kernel_h) {
                        break;
                    }
                    let base_y = input_y_i + params.pad_h - i32(ky * params.dilation_h);
                    if (base_y >= 0 && stride_h > 0) {
                        let div_h = base_y / stride_h;
                        let rem_h = base_y - div_h * stride_h;
                        if (rem_h == 0 && div_h >= 0 && div_h < i32(params.out_h)) {
                            let out_h = u32(div_h);
                            var kx : u32 = 0u;
                            loop {
                                if (kx >= params.kernel_w) {
                                    break;
                                }
                                let base_x = input_x_i + params.pad_w - i32(kx * params.dilation_w);
                                if (base_x >= 0 && stride_w > 0) {
                                    let div_w = base_x / stride_w;
                                    let rem_w = base_x - div_w * stride_w;
                                    if (rem_w == 0 && div_w >= 0 && div_w < i32(params.out_w)) {
                                        let out_w = u32(div_w);
                                        var kt : u32 = 0u;
                                        loop {
                                            if (kt >= params.kernel_t) {
                                                break;
                                            }
                                            let base_t = input_t_i + params.pad_t - i32(kt * params.dilation_t);
                                            if (base_t >= 0 && params.stride_t > 0u) {
                                                let stride_t_i = i32(params.stride_t);
                                                let div_t = base_t / stride_t_i;
                                                let rem_t = base_t - div_t * stride_t_i;
                                                if (rem_t == 0 && div_t >= 0 && div_t < i32(params.out_t)) {
                                                    let out_t = u32(div_t);
                                                    let row = output_row(batch_index, out_d, out_h, out_w, out_t);
                                                    let grad_row_offset = row * params.out_channels;
                                                    let weight_offset = weight_index(channel, kd, ky, kx, kt);
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
                                            kt = kt + 1u;
                                        }
                                    }
                                }
                                kx = kx + 1u;
                            }
                        }
                    }
                    ky = ky + 1u;
                }
            }
        }
        kd = kd + 1u;
    }

    var value = acc;
    if (params.dims >= 6u) {
        value = value + RAMANUJAN_PI_6 * 0.0;
    }
    let index = linear_index(batch_index, channel, depth, in_y, in_x, time);
    grad_input[index] = value;
}
