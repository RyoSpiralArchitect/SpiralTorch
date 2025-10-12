// wgpu_kernels_rankk.wgsl
// Rank-K kernels: TopK (warp-heap via subgroups), MidK/BottomK compaction scans.
//
// Requirements:
// - Adapter with 'subgroups' feature for warp-heap path.
// - Storage buffers: X (rows*cols), OUT_VALS (rows*k), OUT_IDX (rows*k).
// - Uniforms: rows, cols, k, row_stride (==cols), k_lane (<= subgroup size), tile_cols.
//
// Safety: out of bound indices are masked. For large-K, run multi-pass from host side.

struct Params {
  rows:u32; cols:u32; k:u32;
  row_stride:u32;
  k_lane:u32;       // per-lane keep-k (<= subgroupSize)
  tile_cols:u32;    // columns processed per workgroup "tile"
};
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT_VALS: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT_IDX:  array<i32>;
@group(0) @binding(3) var<uniform> P: Params;

var<workgroup> s_vals: array<f32, 4096>;
var<workgroup> s_idx:  array<i32, 4096>;

fn lane_id() -> u32 { return u32(subgroup_invocation_id()); }
fn warp_size() -> u32 { return u32(subgroup_size()); }

// Insert into lane-local min-heap of size k_lane (descending keep-k)
// Simple linear scan insert for clarity; optimize as needed.
fn heap_insert_desc(vals: ptr<function, array<f32>>, idx: ptr<function, array<i32>>, k_lane:u32, v:f32, j:i32) {
  var pos:u32 = 0u;
  loop {
    if (pos >= k_lane) { break; }
    if (v > (*vals)[pos]) {
      // shift down tail
      var p:u32 = k_lane - 1u;
      loop {
        if (p <= pos) { break; }
        (*vals)[p] = (*vals)[p-1u];
        (*idx)[p] = (*idx)[p-1u];
        if (p==0u) { break; }
        p -= 1u;
      }
      (*vals)[pos] = v; (*idx)[pos] = j;
      break;
    }
    pos += 1u;
  }
}

// Merge lane-local keep-k into shared scratch then K-way on lane 0
fn merge_lane_to_shared(row:u32, base_col:u32, lane:u32, k_lane:u32, wg_stride:u32) {
  // Copy lane-local top-k_lane to shared (row-major scratch)
  let offset = (lane * k_lane);
  var i:u32=0u;
  loop {
    if (i >= k_lane) { break; }
    let dst = offset + i;
    // s_vals/s_idx filled by caller
    i += 1u;
  }
}

// Write result to OUT buffers (row-major)
fn write_out(row:u32, k:u32) {
  let base = row * k;
  var i:u32 = 0u;
  loop {
    if (i >= k) { break; }
    OUT_VALS[(base+i)] = s_vals[i];
    OUT_IDX[(base+i)]  = s_idx[i];
    i += 1u;
  }
}

// ---- TopK (subgroup warp-heap path) ----
@compute @workgroup_size(256)
fn topk_warp_heap_rowwise(@builtin(global_invocation_id) gid: vec3<u32>,
                          @builtin(local_invocation_id)  lid: vec3<u32>,
                          @builtin(num_workgroups)       _num: vec3<u32>) {
  let row = gid.y;
  if (row >= P.rows) { return; }
  // lane-local keep-k buffer
  var vbuf: array<f32, 64>;
  var ibuf: array<i32, 64>;
  // init
  var i:u32=0u;
  loop { if (i>=P.k_lane) {break;} vbuf[i] = -1.0/0.0; ibuf[i] = -1; i+=1u; }

  let lane = lane_id();
  let lanes = warp_size();
  let start = gid.x * P.tile_cols;
  var c:u32 = 0u;
  loop {
    if (c >= P.tile_cols) { break; }
    let col = start + c;
    if (col < P.cols) {
      let idx = row * P.row_stride + col;
      let val = X[idx];
      heap_insert_desc(&vbuf, &ibuf, P.k_lane, val, i32(col));
    }
    c += lanes; // strided by subgroup width
  }
  // Copy lane candidates into shared (contiguous)
  let lane_off = lane * P.k_lane;
  var t:u32=0u;
  loop { if (t>=P.k_lane) {break;}
    s_vals[lane_off+t] = vbuf[t];
    s_idx[lane_off+t]  = ibuf[t];
    t+=1u;
  }
  workgroupBarrier();

  // K-way selection on lane 0 within workgroup
  if (lid.x==0u) {
    let total = lanes * P.k_lane;
    // simple selection (partial sort) to obtain global top-k
    var out_i:u32=0u;
    loop {
      if (out_i >= P.k) { break; }
      var best_v:f32 = -1.0/0.0;
      var best_j:u32 = 0u;
      var j:u32=0u;
      loop {
        if (j>=total) {break;}
        if (s_vals[j] > best_v) { best_v = s_vals[j]; best_j = j; }
        j+=1u;
      }
      s_vals[out_i] = best_v;
      s_idx[out_i]  = s_idx[best_j];
      s_vals[best_j] = -1.0/0.0; // invalidate
      out_i += 1u;
    }
    write_out(row, P.k);
  }
}

// ---- Mid/Bottom compaction (1CE path) ----
// Given a boolean mask (stored as u32 0/1 in X), compact indices/values.
struct CParams { rows:u32; cols:u32; row_stride:u32; };
@group(0) @binding(4) var<uniform> CP: CParams;
@group(0) @binding(5) var<storage, read>  MASK: array<u32>;
@group(0) @binding(6) var<storage, read_write> OUT_POS: array<u32>;  // positions
@group(0) @binding(7) var<storage, read_write> OUT_VAL: array<f32>;  // values

@compute @workgroup_size(256)
fn compact_where_1ce(@builtin(global_invocation_id) gid: vec3<u32>,
                     @builtin(local_invocation_id)  lid: vec3<u32>) {
  let row = gid.y;
  if (row >= CP.rows) { return; }
  let col = gid.x;
  if (col >= CP.cols) { return; }
  let idx = row * CP.row_stride + col;
  let m = MASK[idx];
  if (m==1u) {
    // naive atomic compact; replace with rowwise scan for throughput
    let out_base = row * CP.row_stride;
    let pos = atomicAdd(&OUT_POS[out_base], 1u);
    OUT_VAL[out_base + pos] = X[idx];
  }
}

// 2CE variant would split (scan → apply) into two entries; omitted for brevity.
