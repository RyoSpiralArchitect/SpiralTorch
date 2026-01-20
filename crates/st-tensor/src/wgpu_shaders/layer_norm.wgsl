// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct LayerNormParams {
    rows: u32,
    cols: u32,
    flags: u32,
    _pad0: u32,
    epsilon: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> residual : array<f32>;
@group(0) @binding(2) var<storage, read> gamma : array<f32>;
@group(0) @binding(3) var<storage, read> beta : array<f32>;
@group(0) @binding(4) var<storage, read_write> output : array<f32>;
@group(0) @binding(5) var<uniform> params : LayerNormParams;

const FLAG_USE_RESIDUAL: u32 = 1u;
const WG_SIZE: u32 = 256u;

var<workgroup> wg_sum: array<f32, WG_SIZE>;
var<workgroup> wg_sumsq: array<f32, WG_SIZE>;
var<workgroup> wg_mean: f32;
var<workgroup> wg_inv_std: f32;

@compute @workgroup_size(WG_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.rows) {
        return;
    }

    let cols = params.cols;
    let base = row * cols;
    let use_residual = (params.flags & FLAG_USE_RESIDUAL) != 0u;

    var sum: f32 = 0.0;
    var sumsq: f32 = 0.0;

    var c: u32 = lid.x;
    loop {
        if (c >= cols) {
            break;
        }
        let idx = base + c;
        var v = input[idx];
        if (use_residual) {
            v = v + residual[idx];
        }
        sum = sum + v;
        sumsq = sumsq + v * v;
        c = c + WG_SIZE;
    }

    wg_sum[lid.x] = sum;
    wg_sumsq[lid.x] = sumsq;
    workgroupBarrier();

    var stride: u32 = WG_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            wg_sum[lid.x] = wg_sum[lid.x] + wg_sum[lid.x + stride];
            wg_sumsq[lid.x] = wg_sumsq[lid.x] + wg_sumsq[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        let denom = f32(cols);
        let mean = wg_sum[0] / denom;
        let e_x2 = wg_sumsq[0] / denom;
        var var_val = e_x2 - mean * mean;
        var_val = max(var_val, 0.0);
        wg_mean = mean;
        wg_inv_std = inverseSqrt(var_val + params.epsilon);
    }

    workgroupBarrier();
    let mean = wg_mean;
    let inv_std = wg_inv_std;

    var c2: u32 = lid.x;
    loop {
        if (c2 >= cols) {
            break;
        }
        let idx = base + c2;
        var v = input[idx];
        if (use_residual) {
            v = v + residual[idx];
        }
        let normed = (v - mean) * inv_std;
        output[idx] = normed * gamma[c2] + beta[c2];
        c2 = c2 + WG_SIZE;
    }
}
