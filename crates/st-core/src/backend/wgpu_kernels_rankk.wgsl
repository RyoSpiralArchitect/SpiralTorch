// wgpu_kernels_rankk.wgsl (v1.8.2)
//
// Additions over v1.8.0:
// - TopK 1CE: keep‑k heap for small K (<=32) on SUBGROUP path
// - MidK/BottomK: SUBGROUPS apply path uses per‑subgroup base array to reduce atomics/fences

// ===== Common structs/bindings =====
struct Params {
  rows: u32, cols: u32, k: u32,
  row_stride: u32, k_lane: u32, tile_cols: u32,
  radix: u32, segments: u32,
};
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUTV: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUTI: array<u32>;
@group(0) @binding(3) var<uniform> P: Params;

struct CParams {
  rows: u32, cols: u32, row_stride: u32, kind: u32, tiles_x: u32, _pad: u32, _pad2: u32, _pad3: u32,
};
@group(0) @binding(0) var<storage, read>  CX: array<f32>;
@group(0) @binding(1) var<storage, read>  CMASK: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUTPOS: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> OUTVAL: array<f32>;
@group(0) @binding(4) var<uniform> CP: CParams;
@group(0) @binding(5) var<storage, read_write> PREFIX: array<u32>;

// ===== Shared (bank padding) =====
var<workgroup> s_vals: array<f32, 256u + 8u>;
var<workgroup> s_idx:  array<u32, 256u + 8u>;
fn pad_addr(i:u32)->u32 { return i + (i >> 5u); } // +1 per 32

fn next_pow2(x:u32)->u32 {
  var v = max(1u, x - 1u);
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

// ===== TopK 1CE (SUBGROUPS): keep‑k heap for small K =====
const HEAP_K_MAX : u32 = 32u;

fn heap_sift_down(hv: ptr<function, array<f32, 64u>>, hi: ptr<function, array<u32, 64u>>, k:u32, i0:u32) {
  var i = i0;
  loop {
    let l = 2u*i + 1u;
    if (l >= k) { break; }
    var s = l;
    let r = l + 1u;
    if (r < k && (*hv)[r] < (*hv)[l]) { s = r; }
    if ((*hv)[i] <= (*hv)[s]) { break; }
    let tv = (*hv)[i]; let ti = (*hi)[i];
    (*hv)[i] = (*hv)[s]; (*hi)[i] = (*hi)[s];
    (*hv)[s] = tv;       (*hi)[s] = ti;
    i = s;
  }
}

@compute @workgroup_size(256)
fn topk_subgroups_heap_1ce(
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>,
  @builtin(subgroup_size)        sg_size: u32
){
  let r = wid.y;
  if (r >= P.rows) { return; }
  // Phase 0: each lane scans strided to get a candidate
  var best_v: f32 = -0x1p127f;
  var best_i: u32 = 0u;
  var c: u32 = lid.x;
  loop {
    if (c >= P.cols) { break; }
    let v = X[r * P.row_stride + c];
    if (v > best_v) { best_v = v; best_i = c; }
    c += 256u;
  }
  s_vals[pad_addr(lid.x)] = best_v;
  s_idx[pad_addr(lid.x)]  = best_i;
  workgroupBarrier();

  // Phase 1: lane0 builds min‑heap of size K from 256 candidates
  if (lid.x == 0u) {
    let k = min(P.k, HEAP_K_MAX);
    // local heap storage
    var hv: array<f32, 64u>;
    var hi: array<u32, 64u>;
    // init heap with first k candidates
    for (var i:u32=0u; i<k; i=i+1u) {
      hv[i] = s_vals[pad_addr(i)];
      hi[i] = s_idx[pad_addr(i)];
    }
    // build heap
    var i0 = (k / 2u);
    loop { if (i0 == 0u) { break; } i0 = i0 - 1u; heap_sift_down(&hv, &hi, k, i0); }
    // consume remaining candidates
    for (var j:u32=k; j<256u; j=j+1u) {
      let v = s_vals[pad_addr(j)];
      if (v > hv[0]) {
        hv[0] = v; hi[0] = s_idx[pad_addr(j)];
        heap_sift_down(&hv, &hi, k, 0u);
      }
    }
    // write top‑k in descending order by repeatedly popping root
    for (var t:u32=0u; t<k; t=t+1u) {
      // find max in heap (we stored min‑heap; for small k, do linear scan)
      var mx:f32 = hv[0]; var mi:u32 = 0u;
      for (var q:u32=1u; q<k; q=q+1u) {
        if (hv[q] > mx) { mx = hv[q]; mi = q; }
      }
      OUTV[r*P.k + t] = mx;
      OUTI[r*P.k + t] = hi[mi];
      hv[mi] = -0x1p127f; // remove
    }
  }
  workgroupBarrier();
}

// ===== TopK 1CE (SUBGROUPS): bitonic path (portable, as in v1.8.0) =====
@compute @workgroup_size(256)
fn topk_subgroups_bitonic_1ce(
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>,
  @builtin(subgroup_size)        sg_size: u32,
  @builtin(subgroup_invocation_id) sg_lane: u32
){
  let r = wid.y;
  if (r >= P.rows) { return; }

  // local scan
  var best_v: f32 = -0x1p127f;
  var best_i: u32 = 0u;
  var c: u32 = lid.x;
  loop {
    if (c >= P.cols) { break; }
    let v = X[r * P.row_stride + c];
    if (v > best_v) { best_v = v; best_i = c; }
    c += 256u;
  }
  s_vals[pad_addr(lid.x)] = best_v;
  s_idx[pad_addr(lid.x)]  = best_i;
  workgroupBarrier();

  // Subgroup keep‑k (portable O(sg_size) max‑scan). Replace this block with native subgroup reduce when stable.
  let sgc = 256u / max(sg_size, 1u);
  let sg_base = (lid.x / max(sg_size,1u)) * max(sg_size,1u);
  let keep = min(P.k_lane, max(sg_size,1u));
  for (var t:u32=0u; t<keep; t=t+1u) {
    var vmax = s_vals[pad_addr(sg_base + sg_lane)];
    for (var j:u32=0u; j<max(sg_size,1u); j=j+1u) {
      let vv = s_vals[pad_addr(sg_base + j)];
      if (vv > vmax) { vmax = vv; }
    }
    var chosen:u32 = 0xffffffffu;
    for (var j:u32=0u; j<max(sg_size,1u); j=j+1u) {
      if (s_vals[pad_addr(sg_base + j)] == vmax) { chosen = j; break; }
    }
    if (sg_lane == 0u && chosen != 0xffffffffu) {
      let src = sg_base + chosen;
      let dst = sg_base + t;
      let ai = pad_addr(dst); let aj = pad_addr(src);
      let vtmp = s_vals[ai]; let itmp = s_idx[ai];
      s_vals[ai] = s_vals[aj]; s_idx[ai] = s_idx[aj];
      s_vals[aj] = -0x1p127f;  s_idx[aj] = 0u;
    }
    workgroupBarrier();
  }

  // gather subgroup fronts into head [0..cand), pad to pow2 and bitonic sort (desc)
  let cand = sgc * keep;
  if (lid.x == 0u) {
    var t: u32 = 0u;
    for (var k: u32=0u; k<keep; k=k+1u) {
      for (var s: u32=0u; s<sgc; s=s+1u) {
        let src = s*max(sg_size,1u) + k;
        s_vals[pad_addr(t)] = s_vals[pad_addr(src)];
        s_idx[pad_addr(t)]  = s_idx[pad_addr(src)];
        t = t + 1u;
      }
    }
    let n = next_pow2(max(1u,cand));
    for (var i:cand; i<n; i=i+1u) {
      s_vals[pad_addr(i)] = -0x1p127f; s_idx[pad_addr(i)] = 0u;
    }
  }
  workgroupBarrier();

  let n = next_pow2(max(1u, sgc*keep));
  var k2 = 2u;
  loop {
    if (k2 > n) { break; }
    var j = k2 >> 1u;
    loop {
      if (j == 0u) { break; }
      let ixj = lid.x ^ j;
      if (lid.x < n && ixj < n && ixj > lid.x) {
        let up = ((lid.x & k2) == 0u);
        let pa = pad_addr(lid.x); let pb = pad_addr(ixj);
        let va = s_vals[pa]; let vb = s_vals[pb];
        if (up) {
          if (va < vb) { s_vals[pa] = vb; s_vals[pb] = va; let ia = s_idx[pa]; s_idx[pa] = s_idx[pb]; s_idx[pb] = ia; }
        } else {
          if (va > vb) { s_vals[pa] = vb; s_vals[pb] = va; let ia = s_idx[pa]; s_idx[pa] = s_idx[pb]; s_idx[pb] = ia; }
        }
      }
      workgroupBarrier();
      j = j >> 1u;
    }
    k2 = k2 << 1u;
  }
  workgroupBarrier();
  if (lid.x == 0u) {
    let m = min(P.k, n);
    for (var t:u32=0u; t<m; t=t+1u) {
      OUTV[r*P.k + t] = s_vals[pad_addr(t)];
      OUTI[r*P.k + t] = s_idx[pad_addr(t)];
    }
  }
}

// ===== TopK 1CE: workgroup fallback (portable) =====
@compute @workgroup_size(256)
fn topk_workgroup_1ce(
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>
){
  let r = wid.y;
  if (r >= P.rows) { return; }
  var best_v: f32 = -0x1p127f;
  var best_i: u32 = 0u;
  var c: u32 = lid.x;
  loop {
    if (c >= P.cols) { break; }
    let v = X[r * P.row_stride + c];
    if (v > best_v) { best_v = v; best_i = c; }
    c += 256u;
  }
  s_vals[pad_addr(lid.x)] = best_v;
  s_idx[pad_addr(lid.x)]  = best_i;
  workgroupBarrier();

  if (lid.x == 0u) {
    for (var t:u32=0u; t<P.k; t=t+1u) {
      var bi:u32 = 0u; var bv:f32 = -0x1p127f;
      for (var j:u32=0u; j<256u; j=j+1u) {
        let vv = s_vals[pad_addr(j)];
        if (vv > bv) { bv = vv; bi = j; }
      }
      OUTV[r*P.k + t] = s_vals[pad_addr(bi)];
      OUTI[r*P.k + t] = s_idx[pad_addr(bi)];
      s_vals[pad_addr(bi)] = -0x1p127f;
    }
  }
}

// ===== MidK/BottomK: scan tiles (unchanged) =====
var<workgroup> temp: array<u32, 256u>;
@compute @workgroup_size(256)
fn midk_compact_scan_tiles(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
){
  let r = wid.y;
  let tile = gid.x;
  if (r>=CP.rows || tile>=CP.tiles_x) { return; }
  let start = tile * 256u;
  var v: u32 = 0u;
  var c = lid.x;
  loop {
    let col = start + c;
    if (col >= CP.cols) { break; }
    if (CMASK[r*CP.row_stride + col] != 0u) { v += 1u; }
    c += 256u;
  }
  temp[lid.x] = v; workgroupBarrier();

  // Blelloch inclusive total
  var offset = 1u; var d = 256u;
  loop { if (d<=1u){break;} let half=d>>1u;
    if (lid.x < half) {
      let ai = offset*(2u*lid.x + 1u) - 1u;
      let bi = offset*(2u*lid.x + 2u) - 1u;
      temp[bi] = temp[bi] + temp[ai];
    }
    offset = offset << 1u; d = half; workgroupBarrier();
  }
  if (lid.x==0u){ PREFIX[r*CP.tiles_x + tile] = temp[255u]; }
}

// ===== MidK/BottomK: parallel row‑prefix (unchanged from v1.8.0 overlay) =====
@compute @workgroup_size(256)
fn midk_compact_row_prefix(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
){
  let r = wid.y;
  if (r>=CP.rows) { return; }
  var acc: u32 = 0u;
  var base: u32 = 0u;
  loop {
    if (base >= CP.tiles_x) { break; }
    let idx = r*CP.tiles_x + base + lid.x;
    var x: u32 = 0u;
    if (base + lid.x < CP.tiles_x) { x = PREFIX[idx]; }
    temp[lid.x] = x; workgroupBarrier();
    // exclusive in chunk
    var off=1u; var d=256u;
    loop { if (d<=1u){break;} let h=d>>1u;
      if (lid.x < h) {
        let ai = off*(2u*lid.x + 1u) - 1u;
        let bi = off*(2u*lid.x + 2u) - 1u;
        temp[bi] = temp[bi] + temp[ai];
      }
      off = off<<1u; d=h; workgroupBarrier();
    }
    var last = temp[255u];
    if (lid.x == 255u) { temp[255u] = 0u; } workgroupBarrier();
    var h2=128u; var step=128u;
    loop { if (h2==0u){break;}
      if (lid.x < h2) {
        let ai = step*(2u*lid.x + 1u) - 1u;
        let bi = step*(2u*lid.x + 2u) - 1u;
        let t = temp[ai]; temp[ai] = temp[bi]; temp[bi] = temp[bi] + t;
      }
      step = step>>1u; h2 = h2>>1u; workgroupBarrier();
    }
    if (base + lid.x < CP.tiles_x) { PREFIX[idx] = temp[lid.x] + acc; }
    acc = acc + last;
    base = base + 256u;
  }
  if (lid.x == 0u){ atomicStore(&OUTPOS[r], acc); }
}

// ===== MidK/BottomK: SUBGROUP apply (optimized atomics) =====
var<workgroup> wg_sg_base: atomic<u32>;
var<workgroup> sg_bases: array<u32, 8u>; // up to 8 subgroups (256/32)

@compute @workgroup_size(256)
fn midk_compact_apply_sg(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size)        sg_size: u32,
  @builtin(subgroup_invocation_id) sg_lane: u32
){
  let r = wid.y;
  let tile = gid.x;
  if (r>=CP.rows || tile>=CP.tiles_x) { return; }

  let start = tile * 256u;
  let base  = PREFIX[r*CP.tiles_x + tile];

  if (lid.x == 0u) { atomicStore(&wg_sg_base, 0u); }
  workgroupBarrier();

  // lane flags
  var flag: u32 = 0u;
  let col = start + lid.x;
  if (col < CP.cols) { if (CMASK[r*CP.row_stride + col] != 0u) { flag = 1u; } }

  // per-subgroup totals (local count)
  let sgc = 256u / max(sg_size,1u);
  let sg_id = lid.x / max(sg_size,1u);
  var local_excl: u32 = 0u;
  var sg_total:  u32 = 0u;
  // naive per-subgroup exclusive & sum (portable)
  for (var j:u32=0u; j<max(sg_size,1u); j=j+1u) {
    let st = start + j + sg_id*max(sg_size,1u);
    if (st < (start + 256u) && st < CP.cols) {
      let f = u32(CMASK[r*CP.row_stride + st] != 0u);
      if (j < sg_lane) { local_excl = local_excl + f; }
      sg_total = sg_total + f;
    }
  }
  // only lane0 does atomicAdd to reserve subgroup base
  if (sg_lane == 0u) {
    let b = atomicAdd(&wg_sg_base, sg_total);
    sg_bases[sg_id] = b;
  }
  workgroupBarrier();
  let my_base = sg_bases[sg_id];

  // write
  if (flag == 1u && col < CP.cols) {
    let pos = base + my_base + local_excl;
    OUTVAL[r*CP.cols + pos] = CX[r*CP.row_stride + col];
  }
}

// ===== MidK/BottomK: fallback apply (Blelloch) =====
var<workgroup> temp2: array<u32, 256u>;
@compute @workgroup_size(256)
fn midk_compact_apply(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
){
  let r = wid.y;
  let tile = gid.x;
  if (r>=CP.rows || tile>=CP.tiles_x) { return; }
  let start = tile * 256u;
  let base  = PREFIX[r*CP.tiles_x + tile];

  var flag: u32 = 0u;
  let col0 = start + lid.x;
  if (col0 < CP.cols) { if (CMASK[r*CP.row_stride + col0] != 0u) { flag = 1u; } }
  temp2[lid.x] = flag; workgroupBarrier();

  var off=1u; var d=256u;
  loop { if (d<=1u){break;} let h=d>>1u;
    if (lid.x < h) {
      let ai = off*(2u*lid.x + 1u)-1u; let bi = off*(2u*lid.x + 2u)-1u;
      temp2[bi] = temp2[bi] + temp2[ai];
    }
    off=off<<1u; d=h; workgroupBarrier();
  }
  if (lid.x==255u){ temp2[255u]=0u; } workgroupBarrier();
  var h2=128u; var step=128u;
  loop { if (h2==0u){break;}
    if (lid.x < h2) {
      let ai = step*(2u*lid.x + 1u)-1u; let bi = step*(2u*lid.x + 2u)-1u;
      let t = temp2[ai]; temp2[ai]=temp2[bi]; temp2[bi]=temp2[bi]+t;
    }
    step=step>>1u; h2=h2>>1u; workgroupBarrier();
  }
  if (col0 < CP.cols && flag==1u) {
    let pos = base + temp2[lid.x];
    OUTVAL[r*CP.cols + pos] = CX[r*CP.row_stride + col0];
  }
}
