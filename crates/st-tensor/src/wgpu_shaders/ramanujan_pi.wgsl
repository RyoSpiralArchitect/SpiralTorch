// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct RamanujanParams {
    iterations: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read_write> output : array<f32>;
@group(0) @binding(1) var<uniform> params : RamanujanParams;

@compute @workgroup_size(1)
fn main() {
    let iterations = max(params.iterations, 1u);
    var sum : f32 = 0.0;
    var factor : f32 = 1.0;
    let base = pow(396.0, 4.0);
    let prefactor = (2.0 * sqrt(2.0)) / 9801.0;
    var k : u32 = 0u;
    loop {
        if (k >= iterations) {
            break;
        }
        let kf = f32(k);
        sum = sum + factor * (1103.0 + 26390.0 * kf);
        k = k + 1u;
        if (k >= iterations) {
            break;
        }
        let next = f32(k);
        let numerator = (4.0 * next - 3.0)
            * (4.0 * next - 2.0)
            * (4.0 * next - 1.0)
            * (4.0 * next);
        let denominator = pow(next, 4.0) * base;
        factor = factor * (numerator / denominator);
    }
    let denom = prefactor * sum;
    if (denom != 0.0) {
        output[0] = 1.0 / denom;
    } else {
        output[0] = 0.0;
    }
}
