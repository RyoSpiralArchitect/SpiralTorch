struct Params {
  cols: u32,
  rows: u32,
  power: f32,
  pad: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= params.rows || col >= params.cols) {
    return;
  }

  let flat = row * params.cols + col;
  let base = flat * 2u;
  let mirror_col = params.cols - 1u - col;
  let freq = f32(min(col, mirror_col));
  let scale = pow(max(freq, 0.0), params.power);
  y[base] = x[base] * scale;
  y[base + 1u] = x[base + 1u] * scale;
}
