struct RC { rows:u32, cols:u32, k:u32 };
@group(110) @binding(0) var<storage, read>  x: array<f32>;
@group(110) @binding(1) var<storage, read_write> outv: array<f32>;
@group(110) @binding(2) var<storage, read_write> outi: array<u32>;
@group(110) @binding(3) var<uniform> rc: RC;

@compute @workgroup_size(256)
fn topk_pass256(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  if (wid.x >= rc.rows) { return; }
  let cols = rc.cols;
  let K    = rc.k;
  var<workgroup> kv: array<f32, 256u>;
  var<workgroup> ki: array<u32, 256u>;
  if (lid.x == 0u) {
    for (var t:u32=0u; t<K; t=t+1u) { kv[t] = -3.40282347e+38; ki[t] = 0u; }
  }
  workgroupBarrier();
  var j = lid.x;
  loop {
    if (j >= cols) { break; }
    let v = x[wid.x*cols + j];
    if (v > kv[K-1u]) {
      workgroupBarrier();
      if (v > kv[K-1u]) {
        var pos:u32 = K-1u;
        loop {
          if (pos==0u) { break; }
          if (kv[pos-1u] >= v) { break; }
          kv[pos] = kv[pos-1u]; ki[pos] = ki[pos-1u];
          pos = pos - 1u;
        }
        kv[pos] = v; ki[pos] = j;
      }
      workgroupBarrier();
    }
    j = j + 256u;
  }
  if (lid.x == 0u) {
    for (var t:u32=0u; t<K; t=t+1u) {
      let base = wid.x*K + t;
      outv[base] = kv[t];
      outi[base] = ki[t];
    }
  }
}
