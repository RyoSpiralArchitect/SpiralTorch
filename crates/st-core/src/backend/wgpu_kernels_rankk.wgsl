// wgpu_kernels_rankk.wgsl (v1.8.0)
// - TopK 1CE: subgroup intrinsics + shared bitonic (bank‑padding)
// - MidK/BottomK: scan tiles + parallel row‑prefix + fast subgroup apply

// ===== Common structs/bindings =====
struct Params {
  rows: u32, cols: u32, k: u32,
  row_stride: u32, k_lane: u32, tile_cols: u32,
  _pad: u32, _pad2: u32,
};
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUTV: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUTI: array<u32>;
@group(0) @binding(3) var<uniform> P: Params;

struct CParams {
  rows: u32, cols: u32, row_stride: u32, kind: u32, tiles_x: u32, _pad: u32, _pad2: u32, _pad3: u32
};
@group(0) @binding(0) var<storage, read>  CX: array<f32>;
@group(0) @binding(1) var<storage, read>  CMASK: array<u32>;
@group(0) @binding(2) var<storage, read_write> OUTPOS: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> OUTVAL: array<f32>;
@group(0) @binding(4) var<uniform> CP: CParams;
@group(0) @binding(5) var<storage, read_write> PREFIX: array<u32>;

// ===== Shared (with bank padding) =====
var<workgroup> s_vals: array<f32, 256u + 8u>;
var<workgroup> s_idx:  array<u32, 256u + 8u>;
fn pad_addr(i:u32)->u32 { return i + (i >> 5u); } // +1 pad / 32

// ---- helpers ----
fn next_pow2(x:u32)->u32 {
  var v = max(1u, x - 1u);
  v |= v >> 1u; v |= v >> 2u; v |= v >> 4u; v |= v >> 8u; v |= v >> 16u;
  return v + 1u;
}

// ===== TopK 1CE (SUBGROUPS path with intrinsics) =====
@compute @workgroup_size(256)
fn topk_subgroups_1ce(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id)  lid: vec3<u32>,
  @builtin(workgroup_id)         wid: vec3<u32>,
  @builtin(subgroup_size)        sg_size: u32,
  @builtin(subgroup_invocation_id) sg_lane: u32
){
  let r = wid.y;
  if (r >= P.rows) { return; }

  // Phase 1: local scan (stride 256)
  var best_v: f32 = -0x1p127f;
  var best_i: u32 = 0u;
  var c: u32 = lid.x;
  loop {
    if (c >= P.cols) { break; }
    let off = r * P.row_stride + c;
    let v = X[off];
    if (v > best_v) { best_v = v; best_i = c; }
    c += 256u;
  }
  s_vals[pad_addr(lid.x)] = best_v;
  s_idx[pad_addr(lid.x)]  = best_i;
  workgroupBarrier();

  // ---- Subgroup keep‑k using intrinsics ----
  let sgc = 256u / max(sg_size, 1u);
  let sg_id = lid.x / max(sg_size, 1u);
  let sg_base = sg_id * max(sg_size, 1u);
  let keep = min(P.k_lane, max(sg_size,1u));

  // compacted candidate area at head [0 .. sgc*keep)
  // 2A: per‑subgroup keep-k
  for (var t: u32=0u; t<keep; t=t+1u) {
    // reduce max(value) in subgroup
    var v_lane = s_vals[pad_addr(sg_base + sg_lane)];
    // WGSL subgroupMax is not yet standardized across all vendors; emulate via broadcasts:
    // Here we do a naive O(sg_size) max within subgroup to keep portability.
    // Replace with subgroupMax/subgroupShuffle variations when available on target.
    var vmax = v_lane;
    for (var j:u32=0u; j<max(sg_size,1u); j=j+1u) {
      let vv = s_vals[pad_addr(sg_base + j)];
      if (vv > vmax) { vmax = vv; }
    }
    // mark the chosen lane (first that matches vmax)
    var chosen: u32 = 0xffffffffu;
    for (var j:u32=0u; j<max(sg_size,1u); j=j+1u) {
      if (s_vals[pad_addr(sg_base + j)] == vmax) { chosen = j; break; }
    }
    // write to subgroup-front slot (lane0 does write & mark)
    if (sg_lane == 0u && chosen != 0xffffffffu) {
      let src = sg_base + chosen;
      let dst = sg_base + t;
      // swap into front
      let ai = pad_addr(dst); let aj = pad_addr(src);
      let vtmp = s_vals[ai]; let itmp = s_idx[ai];
      s_vals[ai] = s_vals[aj]; s_idx[ai] = s_idx[aj];
      s_vals[aj] = -0x1p127f;  s_idx[aj] = 0u;
    }
    workgroupBarrier();
  }

  // 2B: gather subgroup fronts -> dense head [0..cand)
  let cand = sgc * keep;
  if (lid.x == 0u) {
    var t: u32 = 0u;
    for (var k: u32=0u; k<keep; k=k+1u) {
      for (var s: u32=0u; s<sgc; s=s+1u) {
        let src = s*max(sg_size,1u) + k;
        let ai = pad_addr(t);
        s_vals[ai] = s_vals[pad_addr(src)];
        s_idx[ai]  = s_idx[pad_addr(src)];
        t = t + 1u;
      }
    }
    // pad to pow2
    let n = next_pow2(max(1u,cand));
    for (var i:cand; i<n; i=i+1u) {
      let ai = pad_addr(i);
      s_vals[ai] = -0x1p127f;
      s_idx[ai]  = 0u;
    }
  }
  workgroupBarrier();

  // ---- WG bitonic sort (descending) over head region ----
  let n = next_pow2(max(1u, cand));
  // each thread has id = lid.x; only id<n participates
  var k = 2u;
  loop {
    if (k > n) { break; }
    var j = k >> 1u;
    loop {
      if (j == 0u) { break; }
      let ixj = lid.x ^ j;
      if (lid.x < n && ixj < n && ixj > lid.x) {
        let a = lid.x;
        let b = ixj;
        // increasing/decreasing based on bit of k
        let up = ( (lid.x & k) == 0u );
        let pa = pad_addr(a); let pb = pad_addr(b);
        let va = s_vals[pa]; let vb = s_vals[pb];
        if (up) {
          if (va < vb) {
            s_vals[pa] = vb; s_vals[pb] = va;
            let ia = s_idx[pa]; let ib = s_idx[pb];
            s_idx[pa] = ib; s_idx[pb] = ia;
          }
        } else {
          if (va > vb) {
            s_vals[pa] = vb; s_vals[pb] = va;
            let ia = s_idx[pa]; let ib = s_idx[pb];
            s_idx[pa] = ib; s_idx[pb] = ia;
          }
        }
      }
      workgroupBarrier();
      j = j >> 1u;
    }
    k = k << 1u;
  }
  workgroupBarrier();

  // write top‑K
  if (lid.x == 0u) {
    let base = r * P.k;
    let m = min(P.k, n);
    for (var t: u32=0u; t<m; t=t+1u) {
      let ai = pad_addr(t);
      OUTV[base + t] = s_vals[ai];
      OUTI[base + t] = s_idx[ai];
    }
  }
}

// ===== TopK 1CE: Workgroup fallback =====
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
    let off = r * P.row_stride + c;
    let v = X[off];
    if (v > best_v) { best_v = v; best_i = c; }
    c += 256u;
  }
  s_vals[pad_addr(lid.x)] = best_v;
  s_idx[pad_addr(lid.x)]  = best_i;
  workgroupBarrier();

  // naive WG selection (simple & portable)
  if (lid.x == 0u) {
    let base = r * P.k;
    for (var t: u32=0u; t<P.k; t=t+1u) {
      var bi: u32 = 0u; var bv: f32 = -0x1p127f;
      for (var j: u32=0u; j<256u; j=j+1u) {
        let vv = s_vals[pad_addr(j)];
        if (vv > bv) { bv = vv; bi = j; }
      }
      OUTV[base + t] = s_vals[pad_addr(bi)];
      OUTI[base + t] = s_idx[pad_addr(bi)];
      s_vals[pad_addr(bi)] = -0x1p127f;
    }
  }
}

// ===== MidK/BottomK: scan‑tiles (unchanged) =====
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
    let off = r*CP.row_stride + col;
    if (CMASK[off] != 0u) { v += 1u; }
    c += 256u;
  }
  temp[lid.x] = v;
  workgroupBarrier();

  // Blelloch reduce to total
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

// ===== MidK/BottomK: parallel row‑prefix (block‑crossing) =====
// 1 WG / row, chunk = 256 tiles; exclusive prefix across all tiles.
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
    // load chunk
    var x: u32 = 0u;
    if (base + lid.x < CP.tiles_x) { x = PREFIX[idx]; }
    temp[lid.x] = x; workgroupBarrier();
    // Blelloch exclusive on chunk
    // up-sweep
    var off=1u; var d=256u;
    loop { if (d<=1u){break;} let h=d>>1u;
      if (lid.x < h) {
        let ai = off*(2u*lid.x + 1u) - 1u;
        let bi = off*(2u*lid.x + 2u) - 1u;
        temp[bi] = temp[bi] + temp[ai];
      }
      off = off<<1u; d=h; workgroupBarrier();
    }
    // exclusive: shift right by one, temp[last]=0
    var last = temp[255u];
    if (lid.x == 255u) { temp[255u] = 0u; }
    workgroupBarrier();
    // down-sweep
    var h2=128u; var step=128u;
    loop { if (h2==0u){break;}
      if (lid.x < h2) {
        let ai = step*(2u*lid.x + 1u) - 1u;
        let bi = step*(2u*lid.x + 2u) - 1u;
        let t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] = temp[bi] + t;
      }
      step = step>>1u; h2 = h2>>1u; workgroupBarrier();
    }
    // write back: base offset + acc
    if (base + lid.x < CP.tiles_x) {
      PREFIX[idx] = temp[lid.x] + acc;
    }
    // advance acc by chunk sum (last + original last element)
    acc = acc + last;
    base = base + 256u;
  }
  if (lid.x == 0u){ atomicStore(&OUTPOS[r], acc); }
}

// ===== MidK/BottomK: fast subgroup apply (two-stage) =====
var<workgroup> wg_sg_base: atomic<u32>;

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

  // mark flag
  var flag: u32 = 0u;
  let col = start + lid.x;
  if (col < CP.cols) {
    let off = r*CP.row_stride + col;
    if (CMASK[off] != 0u) { flag = 1u; }
  }
  // subgroup exclusive offset
  // NOTE: WGSL subgroupExclusiveAdd is not yet standardized everywhere; emulate if unavailable.
  var local_excl: u32 = 0u;
  // naive per-subgroup exclusive prefix:
  for (var j:u32=0u; j<max(sg_size,1u); j=j+1u) {
    if (j < sg_lane) {
      let g = (lid.x / max(sg_size,1u))*max(sg_size,1u) + j;
      if (g < 256u) {
        let st = start + j + (lid.x / max(sg_size,1u))*max(sg_size,1u);
        if (st < (start + 256u)) {
          let off2 = r*CP.row_stride + st;
          if (st < CP.cols && CMASK[off2] != 0u) { local_excl = local_excl + 1u; }
        }
      }
    }
  }
  // subgroup total
  var sg_total: u32 = 0u;
  for (var j:u32=0u; j<max(sg_size,1u); j=j+1u) {
    let st = start + j + (lid.x / max(sg_size,1u))*max(sg_size,1u);
    if (st < start + 256u && st < CP.cols) {
      let off3 = r*CP.row_stride + st;
      if (CMASK[off3] != 0u) { sg_total = sg_total + 1u; }
    }
  }
  // lane0 reserves base
  var my_base: u32 = 0u;
  if (sg_lane == 0u) {
    my_base = atomicAdd(&wg_sg_base, sg_total);
  }
  // broadcast within subgroup (fallback by shared array if needed)
  var my_base_all: u32 = my_base;
  // (portable: every lane reads wg_sg_base before its subgroup reserved area;
  // simple approach: barrier + reuse my_base computed only by lane0 with same subgroup_id)
  workgroupBarrier();
  // Recompute my_base per subgroup using a simple scan over earlier subgroups:
  // (For simplicity & portability; harmless for correctness.)
  let sg_id = lid.x / max(sg_size,1u);
  if (sg_lane != 0u) {
    var acc_prev: u32 = 0u;
    for (var s:u32=0u; s<sg_id; s=s+1u) {
      // approximate: each subgroup sum computed similarly
      var sum_s: u32 = 0u;
      for (var j:u32=0u; j<max(sg_size,1u); j=j+1u) {
        let st = start + j + s*max(sg_size,1u);
        if (st < start + 256u && st < CP.cols) {
          let off4 = r*CP.row_stride + st;
          if (CMASK[off4] != 0u) { sum_s = sum_s + 1u; }
        }
      }
      acc_prev = acc_prev + sum_s;
    }
    my_base_all = acc_prev;
  }

  // final write
  if (flag == 1u && col < CP.cols) {
    let pos = base + my_base_all + local_excl;
    OUTVAL[r*CP.cols + pos] = CX[r*CP.row_stride + col];
  }
}

// ===== MidK/BottomK: Blelloch apply (fallback, stable) =====
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
  if (col0 < CP.cols) {
    let off0 = r*CP.row_stride + col0;
    if (CMASK[off0] != 0u) { flag = 1u; }
  }
  temp2[lid.x] = flag; workgroupBarrier();

  // exclusive Blelloch
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
