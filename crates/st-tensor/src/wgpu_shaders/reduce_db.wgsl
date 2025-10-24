// reduce_db.wgsl
// Second pass reduction for GELU bias gradients.

const WG_COLS : u32 = {WG_COLS}u;
const REDUCE_WG : u32 = {REDUCE_WG}u;

struct ReduceUniforms {
  O: u32,
  num_wg_x: u32,
  num_wg_y: u32,
};

@group(0) @binding(0) var<storage, read>       db_partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> db:          array<f32>;
@group(0) @binding(2) var<uniform>             U: ReduceUniforms;

@compute @workgroup_size(REDUCE_WG, 1, 1)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= U.O) {
    return;
  }

  let tile_x = c / WG_COLS;
  let col_in_tile = c % WG_COLS;

  var sum: f32 = 0.0;
  for (var ty: u32 = 0u; ty < U.num_wg_y; ty = ty + 1u) {
    let wg_linear = ty * U.num_wg_x + tile_x;
    let idx = wg_linear * WG_COLS + col_in_tile;
    sum = sum + db_partials[idx];
  }
  db[c] = sum;
}
