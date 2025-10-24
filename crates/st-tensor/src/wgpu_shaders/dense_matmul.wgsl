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

override TILE_SIZE : u32 = {tile_size}u;

var<workgroup> lhs_tile : array<f32, TILE_SIZE * TILE_SIZE>;
var<workgroup> rhs_tile : array<f32, TILE_SIZE * TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.rows || col >= params.cols) {
        return;
    }

    let local_row = lid.y;
    let local_col = lid.x;

    var acc : f32 = 0.0;
    let tiles = (params.inner + TILE_SIZE - 1u) / TILE_SIZE;
    var tile_index : u32 = 0u;
    loop {
        if (tile_index >= tiles) {
            break;
        }

        let k_base = tile_index * TILE_SIZE;

        var lhs_value : f32 = 0.0;
        let lhs_k = k_base + local_col;
        if (lhs_k < params.inner) {
            lhs_value = lhs[row * params.inner + lhs_k];
        }
        lhs_tile[local_row * TILE_SIZE + local_col] = lhs_value;

        var rhs_value : f32 = 0.0;
        let rhs_k = k_base + local_row;
        if (rhs_k < params.inner) {
            rhs_value = rhs[rhs_k * params.cols + col];
        }
        rhs_tile[local_row * TILE_SIZE + local_col] = rhs_value;

        workgroupBarrier();

        var k : u32 = 0u;
        loop {
            if (k >= TILE_SIZE) {
                break;
            }
            let lhs_tile_value = lhs_tile[local_row * TILE_SIZE + k];
            let rhs_tile_value = rhs_tile[k * TILE_SIZE + local_col];
            acc = acc + lhs_tile_value * rhs_tile_value;
            k = k + 1u;
        }

        workgroupBarrier();
        tile_index = tile_index + 1u;
    }

    let out_index = row * params.cols + col;
    out[out_index] = acc;
}
