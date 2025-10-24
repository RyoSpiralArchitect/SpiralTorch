// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

// Workgroup tile dimensions are overridable specialisation constants so
// device-specific dispatch tuning can happen without regenerating the shader.
override WG_ROWS : u32 = 16u;
override WG_COLS : u32 = 16u;

struct Uniforms {
  B: u32;
  O: u32;
  stride: u32;
  num_wg_x: u32;
  num_wg_y: u32;
  add_dR: u32;
};

@group(0) @binding(0) var<storage, read>       Z:           array<f32>;
@group(0) @binding(1) var<storage, read>       G:           array<f32>;
@group(0) @binding(2) var<storage, read_write> gZ_out:      array<f32>;
@group(0) @binding(3) var<storage, read_write> dR_buf:      array<f32>;
@group(0) @binding(4) var<storage, read_write> db_partials: array<f32>;
@group(0) @binding(5) var<uniform>             U: Uniforms;

fn gelu_prime(z: f32) -> f32 {
  let k0: f32 = 0.7978845608028654;
  let k1: f32 = 0.044715;
  let z2 = z * z;
  let u = k0 * (z + k1 * z * z2);
  let t = tanh(u);
  return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t * t) * k0 * (1.0 + 3.0 * k1 * z2);
}

@compute @workgroup_size(WG_COLS, WG_ROWS, 1)
fn main(
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>,
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let base_row = wid.y * WG_ROWS;
  let base_col = wid.x * WG_COLS;

  let row = base_row + lid.y;
  let col = base_col + lid.x;

  var<workgroup> col_sum: array<f32, WG_COLS>;

  if (lid.y == 0u && lid.x < WG_COLS) {
    col_sum[lid.x] = 0.0;
  }
  workgroupBarrier();

  if (row < U.B && col < U.O) {
    let idx = row * U.stride + col;
    let z = Z[idx];
    let g = G[idx];

    let gp = gelu_prime(z);
    let gz = g * gp;

    gZ_out[idx] = gz;

    if (U.add_dR != 0u) {
      dR_buf[idx] = dR_buf[idx] + gz;
    } else {
      dR_buf[idx] = gz;
    }

    if (lid.y == 0u) {
      col_sum[lid.x] = col_sum[lid.x] + gz;
    }
  }

  workgroupBarrier();

  if (lid.y == 0u) {
    if (base_col + lid.x < U.O) {
      let wg_linear = wid.y * U.num_wg_x + wid.x;
      let dst = wg_linear * WG_COLS + lid.x;
      db_partials[dst] = col_sum[lid.x];
    }
  }
}
