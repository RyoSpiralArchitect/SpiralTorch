// SPDX-License-Identifier: AGPL-3.0-or-later
// Part of SpiralTorch - Licensed under AGPL-3.0-or-later.

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

@group(0) @binding(0) var<storage, read> input_tensor: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_tensor: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> params: ConvGemmParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let output_spatial = params.out_h * params.out_w;
    let output_cols = params.out_channels * output_spatial;
    let index = gid.x;
    if (index >= params.batch * output_cols) {
        return;
    }

    let batch_index = index / output_cols;
    let channel = (index / output_spatial) % params.out_channels;
    let position = index % output_spatial;
    let out_y = position / params.out_w;
    let out_x = position % params.out_w;
    var value = bias[channel];

    for (var ky = 0u; ky < params.kernel_h; ky = ky + 1u) {
        for (var kx = 0u; kx < params.kernel_w; kx = kx + 1u) {
            let in_y = i32(out_y * params.stride_h + ky * params.dilation_h) - params.pad_h;
            let in_x = i32(out_x * params.stride_w + kx * params.dilation_w) - params.pad_w;
            if (in_y >= 0 && in_y < i32(params.input_h)
                && in_x >= 0 && in_x < i32(params.input_w)) {
                let input_index = ((batch_index * params.in_channels + channel)
                    * params.input_h + u32(in_y)) * params.input_w + u32(in_x);
                let weight_index = channel * params.span + ky * params.kernel_w + kx;
                value = value + input_tensor[input_index] * weights[weight_index];
            }
        }
    }
    output_tensor[index] = value;
}
