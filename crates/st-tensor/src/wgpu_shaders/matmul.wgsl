// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct Params {
    rows: u32,
    inner: u32,
    cols: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> lhs: array<f32>;

@group(0) @binding(1)
var<storage, read> rhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> out: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 16u;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.rows || col >= params.cols) {
        return;
    }

    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < params.inner; k = k + 1u) {
        let a = lhs[row * params.inner + k];
        let b = rhs[k * params.cols + col];
        acc = acc + a * b;
    }

    out[row * params.cols + col] = acc;
}
