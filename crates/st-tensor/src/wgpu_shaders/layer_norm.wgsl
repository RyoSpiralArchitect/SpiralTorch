// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct LayerNormParams {
    rows: u32,
    cols: u32,
    flags: u32,
    _pad0: u32,
    epsilon_sqrt: f32,
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

var<workgroup> wg_a: array<f32, WG_SIZE>;
var<workgroup> wg_b: array<f32, WG_SIZE>;
var<workgroup> wg_count: array<u32, WG_SIZE>;
var<workgroup> wg_origin: f32;
var<workgroup> wg_scale: f32;
var<workgroup> wg_mean: f32;
var<workgroup> wg_inv_std: f32;

fn value_at(index: u32, use_residual: bool) -> f32 {
    var value = input[index];
    if (use_residual) {
        value = value + residual[index];
    }
    return value;
}

// Avoid an overflowing or subnormal reciprocal in GPU division lowering.
fn divide_scale(value: f32, scale: f32) -> f32 {
    if (value == 0.0) {
        return 0.0;
    }
    if (scale > 1.0e19 || scale < 1.0e-19) {
        let numerator = frexp(value);
        let denominator = frexp(scale);
        return ldexp(numerator.fract / denominator.fract, numerator.exp - denominator.exp);
    }
    return value / scale;
}

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

    // Center at the range midpoint before scaling. Dividing the original
    // values first would discard small, representable differences at large offsets.
    var lo = value_at(base, use_residual);
    var hi = lo;
    var c: u32 = lid.x;
    loop {
        if (c >= cols) {
            break;
        }
        let v = value_at(base + c, use_residual);
        lo = min(lo, v);
        hi = max(hi, v);
        c = c + WG_SIZE;
    }

    wg_a[lid.x] = lo;
    wg_b[lid.x] = hi;
    workgroupBarrier();

    var stride: u32 = WG_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            wg_a[lid.x] = min(wg_a[lid.x], wg_a[lid.x + stride]);
            wg_b[lid.x] = max(wg_b[lid.x], wg_b[lid.x + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lid.x == 0u) {
        var origin = wg_a[0] * 0.5 + wg_b[0] * 0.5;
        if ((wg_a[0] >= 0.0) == (wg_b[0] >= 0.0)) {
            origin = wg_a[0] + (wg_b[0] - wg_a[0]) * 0.5;
        }
        wg_origin = origin;
        wg_scale = max(max(abs(wg_a[0] - origin), abs(wg_b[0] - origin)), params.epsilon_sqrt);
    }
    workgroupBarrier();

    let origin = wg_origin;
    let scale = wg_scale;
    var count = 0u;
    var mean = 0.0;
    var m2 = 0.0;
    c = lid.x;
    loop {
        if (c >= cols) {
            break;
        }
        let value = divide_scale(value_at(base + c, use_residual) - origin, scale);
        count = count + 1u;
        let delta = value - mean;
        mean = mean + delta / f32(count);
        m2 = m2 + delta * (value - mean);
        c = c + WG_SIZE;
    }
    wg_a[lid.x] = mean;
    wg_b[lid.x] = m2;
    wg_count[lid.x] = count;
    workgroupBarrier();

    // Merge Welford moments, never subtract two rounded raw second moments.
    stride = WG_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lid.x < stride) {
            let a = wg_count[lid.x];
            let b = wg_count[lid.x + stride];
            if (a == 0u) {
                wg_a[lid.x] = wg_a[lid.x + stride];
                wg_b[lid.x] = wg_b[lid.x + stride];
            } else if (b != 0u) {
                let delta = wg_a[lid.x + stride] - wg_a[lid.x];
                let weight = f32(b) / f32(a + b);
                wg_a[lid.x] = wg_a[lid.x] + delta * weight;
                wg_b[lid.x] = wg_b[lid.x] + wg_b[lid.x + stride]
                    + delta * delta * f32(a) * weight;
            }
            wg_count[lid.x] = a + b;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        let epsilon_ratio = divide_scale(params.epsilon_sqrt, scale);
        wg_mean = wg_a[0];
        wg_inv_std = inverseSqrt(max(wg_b[0] / f32(cols), 0.0) + epsilon_ratio * epsilon_ratio);
    }

    workgroupBarrier();
    let inv_std = wg_inv_std;

    var c2: u32 = lid.x;
    loop {
        if (c2 >= cols) {
            break;
        }
        let idx = base + c2;
        let v = divide_scale(value_at(idx, use_residual) - origin, scale);
        let normed = (v - wg_mean) * inv_std;
        output[idx] = normed * gamma[c2] + beta[c2];
        c2 = c2 + WG_SIZE;
    }
}
