// wgpu_kernels_rankk.wgsl (reference)
// Single-CE TopK (rowwise). Candidate compression + final K-way in one dispatch.
// NOTE: This reference focuses on IO shape & control. Replace inner loops with your tuned logic.
struct Params {
  rows: u32, cols: u32, k: u32,
  row_stride: u32, k_lane: u32, tile_cols: u32,
  _pad: u32, _pad2: u32,
};
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUTV: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUTI: array<u32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> s_vals: array<f32, 256u>;
var<workgroup> s_idx:  array<u32, 256u>;

@compute @workgroup_size(256)
fn topk_subgroups_1ce(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>
) {
  let r = wid.y;                 // row id
  if (r >= P.rows) { return; }
  // This kernel assumes one WG per row (gx==1). Ensure tile_cols>=cols at dispatch.
  var best_v: f32 = -0x1p127f;   // -inf
  var best_i: u32 = 0u;
  var c: u32 = lid.x;
  // Stride across columns, keep single best per-thread (skeleton)
  loop {
    if (c >= P.cols) { break; }
    let off = r * P.row_stride + c;
    let v = X[off];
    if (v > best_v) { best_v = v; best_i = c; }
    c += 256u;
  }
  s_vals[lid.x] = best_v;
  s_idx[lid.x]  = best_i;
  workgroupBarrier();

  // Final K-way (skeleton): select top-K from 256 candidates
  // For large K, replace with heap/K-way using subgroup/shared.
  if (lid.x == 0u) {
    var used: array<bool, 256u>;
    for (var i: u32=0u; i<256u; i=i+1u) { used[i] = false; }
    let base = r * P.k;
    for (var t: u32=0u; t<P.k; t=t+1u) {
      var bi: u32 = 0u;
      var bv: f32 = -0x1p127f;
      for (var j: u32=0u; j<256u; j=j+1u) {
        if (!used[j] && s_vals[j] > bv) { bv = s_vals[j]; bi = j; }
      }
      used[bi] = true;
      OUTV[base + t] = s_vals[bi];
      OUTI[base + t] = s_idx[bi];
    }
  }
}

// ---- MidK/BottomK compaction ----
struct CParams { rows: u32, cols: u32, row_stride: u32, kind: u32 /*0:mid,1:bottom*/ };
@group(0) @binding(0) var<storage, read>  CX: array<f32>;
@group(0) @binding(1) var<storage, read>  CMASK: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUTPOS: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> OUTVAL: array<f32>;
@group(0) @binding(4) var<uniform> CP: CParams;

// 1CE: atomic compaction (unordered)
@compute @workgroup_size(256)
fn midk_compact_1ce(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let r = wid.y;
  if (r>=CP.rows) { return; }
  var c = lid.x;
  loop {
    if (c>=CP.cols) { break; }
    let off = r*CP.row_stride + c;
    let m = CMASK[off];
    if (m != 0u) {
      let pos = atomicAdd(&OUTPOS[r], 1u);
      OUTVAL[r*CP.cols + pos] = CX[off];
    }
    c += 256u;
  }
}

// 2CE-pass A: per-row scan (exclusive)
var<workgroup> psum: array<u32, 256u>;
@compute @workgroup_size(256)
fn midk_compact_scan_pass(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let r = wid.y;
  if (r>=CP.rows) { return; }
  // Each thread scans a strided chunk; this skeleton performs a WG-local sum only.
  // Replace with tiled Blelloch scan per row (shared) as needed.
  var local_sum: u32 = 0u;
  var c = lid.x;
  loop {
    if (c>=CP.cols) { break; }
    let off = r*CP.row_stride + c;
    let m = CMASK[off];
    if (m!=0u) { local_sum += 1u; }
    c += 256u;
  }
  psum[lid.x] = local_sum;
  workgroupBarrier();
  // parallel reduction
  var step = 128u;
  loop {
    if (step==0u) { break; }
    if (lid.x < step) { psum[lid.x] += psum[lid.x + step]; }
    workgroupBarrier();
    step = step >> 1u;
  }
  if (lid.x==0u) { OUTPOS[r] = psum[0]; } // total trues (rowwise)
}

// 2CE-pass B: apply using OUTPOS(r) as per-row base (skeleton)
@compute @workgroup_size(256)
fn midk_compact_apply_pass(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let r = wid.y;
  if (r>=CP.rows) { return; }
  // This pass would use prefix indices; this reference simply re-uses atomic base.
  var c = lid.x;
  loop {
    if (c>=CP.cols) { break; }
    let off = r*CP.row_stride + c;
    let m = CMASK[off];
    if (m != 0u) {
      let pos = atomicAdd(&OUTPOS[r], 1u);
      OUTVAL[r*CP.cols + pos] = CX[off];
    }
    c += 256u;
  }
}
