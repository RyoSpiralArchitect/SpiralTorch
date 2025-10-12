enable chromium_experimental_subgroups;

struct Meta { rows:u32; cols:u32; k:u32; stride:u32; cand_cols:u32; }
@group(0) @binding(0) var<uniform> U : Meta;
@group(0) @binding(1) var<storage, read>  X : array<f32>;
@group(0) @binding(2) var<storage, read_write> CVAL : array<f32>;
@group(0) @binding(3) var<storage, read_write> CIDX : array<i32>;
@group(0) @binding(4) var<storage, read_write> OVAL : array<f32>;
@group(0) @binding(5) var<storage, read_write> OIDX : array<i32>;

fn idx(row:u32, col:u32, stride:u32)->u32 { return row*stride + col; }

@compute @workgroup_size(256)
fn topk_pass1_subgroup(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  if (row >= U.rows) { return; }
  let lane = subgroupInvocationID();
  let lanes = subgroupSize();
  var best_v: f32 = -1.0/0.0;
  var best_i: i32 = -1;
  var col:u32 = gid.x + lane;
  while (col < U.cols) {
    let v = X[idx(row, col, U.stride)];
    if (v > best_v) { best_v = v; best_i = i32(col); }
    col += lanes;
  }
  // subgroup reduce
  var off:u32 = lanes/2u;
  loop {
    if (off==0u) { break; }
    let ov = subgroupShuffleXor(best_v, off);
    let oi = subgroupShuffleXor(f32(best_i), off);
    if (ov > best_v) { best_v = ov; best_i = i32(oi); }
    off = off/2u;
  }
  if (lane == 0u) {
    let out_base = row * U.cand_cols;
    let out_i = out_base + (gid.x/lanes);
    if (out_i < out_base + U.cand_cols) {
      CVAL[out_i] = best_v;
      CIDX[out_i] = best_i;
    }
  }
}

@compute @workgroup_size(256)
fn topk_pass2_kway_subgroup(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Per row merge: repeated subgroup max over candidate buffer; mark taken by -inf write.
  let row = gid.x;
  if (row >= U.rows) { return; }
  let lane = subgroupInvocationID();
  let lanes = subgroupSize();
  let base = row * U.cand_cols;
  for (var t:u32=0u; t<U.k; t=t+1u) {
    var best_v: f32 = -1.0/0.0;
    var best_j: u32 = 0u;
    var j:u32 = lane;
    while (j < U.cand_cols) {
      let v = CVAL[base+j];
      if (v > best_v) { best_v = v; best_j = j; }
      j += lanes;
    }
    // reduce
    var off:u32 = lanes/2u;
    loop {
      if (off==0u) { break; }
      let ov = subgroupShuffleXor(best_v, off);
      let oj = u32(subgroupShuffleXor(f32(best_j), off));
      if (ov > best_v) { best_v = ov; best_j = oj; }
      off = off/2u;
    }
    if (lane==0u) {
      OVAL[row*U.k + t] = best_v;
      OIDX[row*U.k + t] = CIDX[base + best_j];
      CVAL[base + best_j] = -1.0/0.0; // mark taken
    }
    workgroupBarrier();
  }
}
