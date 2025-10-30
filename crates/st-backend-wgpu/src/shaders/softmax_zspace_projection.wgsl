// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct Params {
    rows: u32,
    cols: u32,
    stride: u32,
    _pad: u32,
    golden_ratio: f32,
    golden_angle: f32,
    min_energy: f32,
    _pad1: f32,
};

@group(0) @binding(0)
var<storage, read> softmax_output: array<f32>;

@group(0) @binding(1)
var<storage, read_write> zspace_metrics: array<vec4<f32>>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.rows) {
        return;
    }

    let cols = params.cols;
    if (cols == 0u) {
        zspace_metrics[row] = vec4<f32>(0.0, 0.33333334, 0.33333334, 0.0);
        return;
    }

    let phi = params.golden_ratio;
    let angle = params.golden_angle;
    var band_above = u32(max(1.0, floor(f32(cols) / phi)));
    var band_here = u32(max(1.0, floor(f32(cols) / (phi * phi))));
    if (band_above >= cols) {
        band_above = cols;
        band_here = 0u;
    } else if (band_above + band_here >= cols) {
        band_here = cols - band_above;
    }
    let band_split = band_above + band_here;

    let base = row * params.stride;
    var focus = 0.0;
    var above = 0.0;
    var here = 0.0;
    var beneath = 0.0;
    var swirl = 0.0;

    for (var c: u32 = 0u; c < cols; c = c + 1u) {
        let value = softmax_output[base + c];
        focus = focus + value * value;
        let phase = f32(c) * angle;
        swirl = swirl + value * sin(phase);
        if (c < band_above) {
            above = above + value;
        } else if (c < band_split) {
            here = here + value;
        } else {
            beneath = beneath + value;
        }
    }

    var energy = above + here + beneath;
    if (energy <= params.min_energy) {
        zspace_metrics[row] = vec4<f32>(0.0, 0.33333334, 0.33333334, 0.0);
        return;
    }

    let inv = 1.0 / energy;
    above = above * inv;
    here = here * inv;
    focus = focus * inv;
    swirl = swirl * inv;

    zspace_metrics[row] = vec4<f32>(focus, above, here, swirl);
}
