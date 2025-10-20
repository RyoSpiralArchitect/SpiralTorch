// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct Im2ColParams {
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
};

@group(0) @binding(0) var<storage, read> input_tensor : array<f32>;
@group(0) @binding(1) var<storage, read_write> patches : array<f32>;
@group(0) @binding(2) var<uniform> params : Im2ColParams;

const WG_X : u32 = 8u;
const WG_Y : u32 = 8u;

fn input_index(batch: u32, channel: u32, y: u32, x: u32) -> u32 {
    let plane = params.input_h * params.input_w;
    return (((batch * params.in_channels) + channel) * plane) + y * params.input_w + x;
}

@compute @workgroup_size(WG_X, WG_Y, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let ow_idx = gid.x;
    let oh_idx = gid.y;
    let batch_idx = gid.z;

    if (batch_idx >= params.batch || oh_idx >= params.out_h || ow_idx >= params.out_w) {
        return;
    }

    let spatial = params.out_h * params.out_w;
    let base = (batch_idx * spatial + oh_idx * params.out_w + ow_idx) * params.span;

    var offset : u32 = 0u;
    for (var ic : u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var kh : u32 = 0u; kh < params.kernel_h; kh = kh + 1u) {
            for (var kw : u32 = 0u; kw < params.kernel_w; kw = kw + 1u) {
                let in_y = i32(oh_idx) * i32(params.stride_h) + i32(kh) * i32(params.dilation_h) - params.pad_h;
                let in_x = i32(ow_idx) * i32(params.stride_w) + i32(kw) * i32(params.dilation_w) - params.pad_w;

                var value : f32 = 0.0;
                if (in_y >= 0 && in_x >= 0 && in_y < i32(params.input_h) && in_x < i32(params.input_w)) {
                    let iy = u32(in_y);
                    let ix = u32(in_x);
                    let idx = input_index(batch_idx, ic, iy, ix);
                    value = input_tensor[idx];
                }

                patches[base + offset] = value;
                offset = offset + 1u;
            }
        }
    }
}
