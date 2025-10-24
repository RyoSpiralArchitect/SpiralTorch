enable chromium_experimental_subgroups;

// Packed input: u64 as 2x u32 (lo=index bits, hi=f32 bits)
struct InP { data: array<u32>, };
struct OutV { data: array<f32>, };
struct OutI { data: array<i32>, };

// P: rows, cols, k_final, tile_cols
struct Params { rows:u32, cols:u32, k:u32, tile:u32, }

@group(0) @binding(0) var<storage, read>       packed : InP;
@group(0) @binding(1) var<storage, read_write> out_v  : OutV;
@group(0) @binding(2) var<storage, read_write> out_i  : OutI;
@group(0) @binding(3) var<uniform>             P      : Params;

fn unpack_pair(off:u32)->vec2<u32> {
  let lo = packed.data[off];
  let hi = packed.data[off+1u];
  return vec2<u32>(lo, hi);
}
fn f32h(x:u32)->f32 { return bitcast<f32>(x); }
fn i32l(x:u32)->i32 { return bitcast<i32>(x); }

// tiny keep‑4 descending
fn insert4_desc(kv: ptr<function, array<f32,4>>, ki: ptr<function, array<i32,4>>, v:f32, ix:i32){
  if (v <= (*kv)[3u]) { return; }
  (*kv)[3u]=v; (*ki)[3u]=ix;
  if ((*kv)[3u] > (*kv)[2u]) { let tv=(*kv)[2u]; let ti=(*ki)[2u]; (*kv)[2u]=(*kv)[3u]; (*ki)[2u]=(*ki)[3u]; (*kv)[3u]=tv; (*ki)[3u]=ti; }
  if ((*kv)[2u] > (*kv)[1u]) { let tv=(*kv)[1u]; let ti=(*ki)[1u]; (*kv)[1u]=(*kv)[2u]; (*ki)[1u]=(*ki)[2u]; (*kv)[2u]=tv; (*ki)[2u]=ti; }
  if ((*kv)[1u] > (*kv)[0u]) { let tv=(*kv)[0u]; let ti=(*ki)[0u]; (*kv)[0u]=(*kv)[1u]; (*ki)[0u]=(*ki)[1u]; (*kv)[1u]=tv; (*ki)[1u]=ti; }
}

var<workgroup> pool_v : array<f32, 2048>;  // pooled leaders (cap)
var<workgroup> pool_i : array<i32, 2048>;
var<workgroup> pool_n : atomic<u32>;

@compute @workgroup_size(256)
fn main_cs(@builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id)       wid: vec3<u32>,
           @builtin(subgroup_size)      sg_sz: u32,
           @builtin(subgroup_invocation_id) sg_lane: u32)
{
  let rows  = P.rows;
  let cols  = P.cols;
  let K     = P.k;
  let TILE  = max(P.tile, 256u);
  let r = wid.x;
  if (r >= rows) { return; }
  if (lid.x == 0u) { atomicStore(&pool_n, 0u); }
  workgroupBarrier();

  // per-thread tiny keep-4
  var kv: array[f32, 4];
  var ki: array[i32, 4];
  for (var j=0u;j<4u;j++){ kv[j]=-1e30; ki[j]=-1; }

  // tiled sweep across columns
  let row_off = r * cols;
  var c0 = 0u;
  loop {
    if (c0 >= cols) { break; }
    let cend = min(c0 + TILE, cols);
    var c = c0 + lid.x;
    while (c < cend) {
      let off = (row_off + c) * 2u;
      let pr  = unpack_pair(off);
      insert4_desc(&kv, &ki, f32h(pr.y), i32l(pr.x));
      c += 256u;
    }
    // subgroup reduction in this tile
    var step = 1u;
    loop {
      if (step >= sg_sz) { break; }
      if ((sg_lane % (2u*step)) == 0u) {
        let partner = sg_lane + step;
        if (partner < sg_sz) {
          var bv: array<f32,4>;
          var bi: array<i32,4>;
          for (var j=0u;j<4u;j++){
            // broadcast from partner
            let pv = subgroupBroadcastFirst(kv[j], partner);
            let pi = subgroupBroadcastFirst(ki[j], partner);
            bv[j]=pv; bi[j]=pi;
          }
          // merge
          var tmpv: array<f32,4>; var tmpi: array<i32,4>;
          for (var j=0u;j<4u;j++){ tmpv[j]=-1e30; tmpi[j]=-1; }
          for (var j=0u;j<4u;j++){ insert4_desc(&tmpv,&tmpi, kv[j], ki[j]); }
          for (var j=0u;j<4u;j++){ insert4_desc(&tmpv,&tmpi, bv[j], bi[j]); }
          for (var j=0u;j<4u;j++){ kv[j]=tmpv[j]; ki[j]=tmpi[j]; }
        }
      }
      step = step << 1u;
    }
    // stage subgroup leaders to pool
    if (sg_lane == 0u) {
      let sgid = (lid.x / sg_sz);
      let base = atomicAdd(&pool_n, 4u);
      if (base < 2048u) { // cap
        for (var j=0u;j<4u;j++){ pool_v[base+j]=kv[j]; pool_i[base+j]=ki[j]; }
      }
    }
    workgroupBarrier();
    // reset per-thread keep-4 for next tile (optional; not necessary if we want sliding selection)
    for (var j=0u;j<4u;j++){ kv[j]=-1e30; ki[j]=-1; }
    c0 += TILE;
  }
  workgroupBarrier();

  // final selection from pooled leaders (size = pool_n)
  if (lid.x == 0u) {
    let pool = min(atomicLoad(&pool_n), 2048u);
    let off = r * K;
    // naive selection (pool should be small after pooling); can replace with small heap if needed
    for (var j=0u; j<K; j++) {
      var best = j;
      for (var t=j+1u; t<pool; t++) {
        if (pool_v[t] > pool_v[best]) { best = t; }
      }
      let tv=pool_v[best]; let ti=pool_i[best];
      pool_v[best]=pool_v[j]; pool_i[best]=pool_i[j];
      pool_v[j]=tv; pool_i[j]=ti;
      out_v.data[off+j]=pool_v[j];
      out_i.data[off+j]=pool_i[j];
    }
  }
}
