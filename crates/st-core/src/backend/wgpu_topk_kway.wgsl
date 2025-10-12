struct Meta { rows:u32, cols:u32, k:u32, k_lane:u32, chunk_cols:u32, cand_cols:u32 };
@group(0) @binding(0) var<storage, read>  X: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUTV: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUTI: array<i32>;
@group(0) @binding(3) var<uniform> meta: Meta;
@group(0) @binding(4) var<storage, read_write> CANDV: array<f32>;
@group(0) @binding(5) var<storage, read_write> CANDI: array<i32>;

var<workgroup> cand_vals: array<f32, 256*32>;
var<workgroup> cand_idxs: array<i32, 256*32>;

fn row_base(r:u32, cols:u32) -> u32 { return r * cols; }

fn scan_and_keep(tid:u32, stride:u32, base:u32, start_col:u32, end_col:u32, k_lane:u32, inout_vals: ptr<function, array<f32,32>>, inout_idxs: ptr<function, array<i32,32>>) {
  var c:u32 = start_col + tid;
  while (c < end_col) {
    let v = X[base + c];
    var minv:f32 = (*inout_vals)[0];
    var minp:u32 = 0u32;
    for (var j:u32=1u32; j<k_lane; j=j+1u32) {
      if ((*inout_vals)[j] < minv) { minv = (*inout_vals)[j]; minp = j; }
    }
    if (v > minv) {
      (*inout_vals)[minp] = v;
      (*inout_idxs)[minp] = i32(c);
    }
    c = c + stride;
  }
}

fn bitonic_desc(total:u32, tid:u32) {
  var size:u32 = 2u;
  while (size <= total) {
    var stride2:u32 = size / 2u;
    while (stride2 > 0u) {
      var i:u32 = tid;
      while (i < total) {
        let j = i ^ stride2;
        if (j > i) {
          let up = ((i & size) == 0u);
          let vi = cand_vals[i];
          let vj = cand_vals[j];
          if ((up && vi < vj) || (!up && vi > vj)) {
            let t = cand_vals[i]; cand_vals[i] = cand_vals[j]; cand_vals[j] = t;
            let ti = cand_idxs[i]; cand_idxs[i] = cand_idxs[j]; cand_idxs[j] = ti;
          }
        }
        i = i + 256u;
      }
      workgroupBarrier();
      stride2 = stride2 / 2u;
    }
    size = size * 2u;
  }
}

fn topk_impl_1ce(wg_size:u32, row:u32, tid:u32) {
  if (row >= meta.rows) { return; }
  let stride = wg_size;
  let k_lane = meta.k_lane;
  let base = row_base(row, meta.cols);

  var local_vals: array<f32, 32>;
  var local_idxs: array<i32, 32>;
  for (var i:u32=0u32; i<k_lane; i=i+1u32) { local_vals[i] = -1.0/0.0; local_idxs[i] = -1; }

  let chunk = select(meta.cols, meta.chunk_cols, meta.chunk_cols!=0u);
  var pos:u32 = 0u;
  loop {
    let start_c = pos * chunk;
    if (start_c >= meta.cols) { break; }
    let end_c = min(start_c + chunk, meta.cols);
    scan_and_keep(tid, stride, base, start_c, end_c, k_lane, &local_vals, &local_idxs);
    pos = pos + 1u;
    if (meta.chunk_cols==0u) { break; }
  }

  let offset = tid * k_lane;
  for (var i:u32=0u32; i<k_lane; i=i+1u32) {
    cand_vals[offset + i] = local_vals[i];
    cand_idxs[offset + i] = local_idxs[i];
  }
  workgroupBarrier();

  let total = 256u * k_lane;
  bitonic_desc(total, tid);

  if (tid < meta.k) {
    OUTV[row*meta.k + tid] = cand_vals[tid];
    OUTI[row*meta.k + tid] = cand_idxs[tid];
  }
}

@compute @workgroup_size(128)
fn topk_kway_1ce_128(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  topk_impl_1ce(128u, wid.x, lid.x);
}
@compute @workgroup_size(256)
fn topk_kway_1ce_256(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  topk_impl_1ce(256u, wid.x, lid.x);
}

fn topk_impl_pass1(wg_size:u32, row:u32, tid:u32) {
  if (row >= meta.rows) { return; }
  let stride = wg_size;
  let k_lane = meta.k_lane;
  let base = row_base(row, meta.cols);

  var local_vals: array<f32, 32>;
  var local_idxs: array<i32, 32>;
  for (var i:u32=0u32; i<k_lane; i=i+1u32) { local_vals[i] = -1.0/0.0; local_idxs[i] = -1; }

  let chunk = select(meta.cols, meta.chunk_cols, meta.chunk_cols!=0u);
  var pos:u32 = 0u;
  loop {
    let start_c = pos * chunk;
    if (start_c >= meta.cols) { break; }
    let end_c = min(start_c + chunk, meta.cols);
    scan_and_keep(tid, stride, base, start_c, end_c, k_lane, &local_vals, &local_idxs);
    pos = pos + 1u;
    if (meta.chunk_cols==0u) { break; }
  }

  let base_cand = row * meta.cand_cols;
  let offset = tid * k_lane;
  for (var i:u32=0u32; i<k_lane; i=i+1u32) {
    CANDV[base_cand + offset + i] = local_vals[i];
    CANDI[base_cand + offset + i] = local_idxs[i];
  }
}
fn topk_impl_pass2(_wg_size:u32, row:u32, tid:u32) {
  if (row >= meta.rows) { return; }
  let k_lane = meta.k_lane;
  let total = 256u * k_lane;
  let base_cand = row * meta.cand_cols;
  let offset = tid * k_lane;
  for (var i:u32=0u32; i<k_lane; i=i+1u32) {
    cand_vals[offset + i] = CANDV[base_cand + offset + i];
    cand_idxs[offset + i] = CANDI[base_cand + offset + i];
  }
  workgroupBarrier();
  bitonic_desc(total, tid);
  if (tid < meta.k) {
    OUTV[row*meta.k + tid] = cand_vals[tid];
    OUTI[row*meta.k + tid] = cand_idxs[tid];
  }
}
@compute @workgroup_size(128)
fn topk_kway_pass1_128(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  topk_impl_pass1(128u, wid.x, lid.x);
}
@compute @workgroup_size(256)
fn topk_kway_pass1_256(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  topk_impl_pass1(256u, wid.x, lid.x);
}
@compute @workgroup_size(128)
fn topk_kway_pass2_128(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  topk_impl_pass2(128u, wid.x, lid.x);
}
@compute @workgroup_size(256)
fn topk_kway_pass2_256(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  topk_impl_pass2(256u, wid.x, lid.x);
}
