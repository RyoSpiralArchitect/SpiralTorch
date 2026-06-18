struct Params {
  cols: u32,
  rows: u32,
  klen: u32,
  alpha_scale: f32,
  pad_mode: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> coeff: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= params.rows) {
    return;
  }

  for (var col: u32 = 0u; col < params.cols; col = col + 1u) {
    var acc = 0.0;
    for (var k: u32 = 0u; k < params.klen; k = k + 1u) {
      var sample = 0.0;
      if (col >= k) {
        sample = x[row * params.cols + (col - k)];
      } else if (params.pad_mode == 1u) {
        sample = x[row * params.cols];
      }
      acc = acc + coeff[k] * sample;
    }
    y[row * params.cols + col] = acc * params.alpha_scale;
  }
}
