struct RC { rows:u32, cols:u32, k:u32, neg_inf:f32 };
@group(111) @binding(0) var<storage, read_write> x: array<f32>;
@group(111) @binding(1) var<storage, read>  idx: array<u32>;
@group(111) @binding(2) var<uniform> rc: RC;
@compute @workgroup_size(256)
fn mask_indices(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  if (wid.x >= rc.rows) { return; }
  var t = lid.x;
  loop {
    if (t >= rc.k) { break; }
    let j = idx[wid.x*rc.k + t];
    if (j < rc.cols) { x[wid.x*rc.cols + j] = rc.neg_inf; }
    t = t + 256u;
  }
}
