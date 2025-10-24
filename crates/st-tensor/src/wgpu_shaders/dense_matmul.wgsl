// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

struct MatmulParams {
    rows: u32,
    cols: u32,
    inner: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> lhs : array<f32>;
@group(0) @binding(1) var<storage, read> rhs : array<f32>;
@group(0) @binding(2) var<storage, read_write> out : array<f32>;
@group(0) @binding(3) var<uniform> params : MatmulParams;

const TILE_X : u32 = 16u;
const TILE_Y : u32 = 16u;
const TILE_K : u32 = 16u;

var<workgroup> tile_a : array<f32, TILE_Y * TILE_K>;
var<workgroup> tile_b : array<f32, TILE_K * TILE_X>;

@compute @workgroup_size(TILE_X, TILE_Y, 1)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    let in_bounds = row < params.rows && col < params.cols;

    var acc : f32 = 0.0;
    let tiles = (params.inner + TILE_K - 1u) / TILE_K;
    var tile_idx : u32 = 0u;
    loop {
        if (tile_idx >= tiles) {
            break;
        }

        let k_base = tile_idx * TILE_K;

        let a_col = k_base + local_col;
        var lhs_value : f32 = 0.0;
        if (row < params.rows && a_col < params.inner) {
            lhs_value = lhs[row * params.inner + a_col];
        }
        tile_a[(local_row * TILE_K) + local_col] = lhs_value;

        let b_row = k_base + local_row;
        var rhs_value : f32 = 0.0;
        if (col < params.cols && b_row < params.inner) {
            rhs_value = rhs[b_row * params.cols + col];
        }
        tile_b[(local_row * TILE_X) + local_col] = rhs_value;

        workgroupBarrier();

        var kk : u32 = 0u;
        loop {
            if (kk >= TILE_K) {
                break;
            }
            acc = acc + tile_a[(local_row * TILE_K) + kk] * tile_b[(kk * TILE_X) + local_col];
            kk = kk + 1u;
        }

        workgroupBarrier();
        tile_idx = tile_idx + 1u;
    }

    if (in_bounds) {
        let out_index = row * params.cols + col;
        out[out_index] = acc;
    }
}
