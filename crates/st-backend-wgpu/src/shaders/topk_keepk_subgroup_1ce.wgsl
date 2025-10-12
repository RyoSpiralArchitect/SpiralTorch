enable chromium_experimental_subgroups;

struct InP { data: array<u32>; };  // u64 packed as 2x u32: [lo(idx), hi(f32_bits)]
struct OutV { data: array<f32>; };
struct OutI { data: array<i32>; };

@group(0) @binding(0) var<storage, read> packed : InP;
@group(0) @binding(1) var<storage, read_write> out_vals : OutV;
@group(0) @binding(2) var<storage, read_write> out_idx  : OutI;

// rows, total, k_final, stride (reserved)
@group(0) @binding(3) var<uniform> P : vec4<u32>;

fn unpack_at(base:u32)->vec2<u32>{
  let lo = packed.data[base];
  let hi = packed.data[base+1];
  return vec2<u32>(lo, hi);
}
fn f32_from_hi(hi:u32)->f32 { return bitcast<f32>(hi); }
fn i32_from_lo(lo:u32)->i32 { return bitcast<i32>(lo); }

// Tiny keep‑k M=4 inserted ascending -> we maintain descending after insertion
fn insert4_desc(mut kv: ptr<function, array<f32,4>>,
                mut ki: ptr<function, array<i32,4>>,
                v:f32, ix:i32){
  if (v <= (*kv)[3u]) { return; }
  (*kv)[3u] = v; (*ki)[3u] = ix;
  if ((*kv)[3u] > (*kv)[2u]) { let tv=(*kv)[2u]; let ti=(*ki)[2u]; (*kv)[2u]=(*kv)[3u]; (*ki)[2u]=(*ki)[3u]; (*kv)[3u]=tv; (*ki)[3u]=ti; }
  if ((*kv)[2u] > (*kv)[1u]) { let tv=(*kv)[1u]; let ti=(*ki)[1u]; (*kv)[1u]=(*kv)[2u]; (*ki)[1u]=(*ki)[2u]; (*kv)[2u]=tv; (*ki)[2u]=ti; }
  if ((*kv)[1u] > (*kv)[0u]) { let tv=(*kv)[0u]; let ti=(*ki)[0u]; (*kv)[0u]=(*kv)[1u]; (*ki)[0u]=(*ki)[1u]; (*kv)[1u]=tv; (*ki)[1u]=ti; }
}

// Merge two 4‑lists to keep top‑4
fn merge4_keep4(a:array<f32,4>, ai:array<i32,4>, b:array<f32,4>, bi:array<i32,4>) -> vec4<f32> {
  var pa:u32 = 0u;
  var pb:u32 = 0u;
  var out: array<f32,4>;
  var outi: array<i32,4>;
  for (var j=0u; j<4u; j++){
    let va = select(-1e30, a[pa], pa<4u);
    let vb = select(-1e30, b[pb], pb<4u);
    let take_a = va >= vb;
    out[j]  = select(vb, va, take_a);
    outi[j] = select(bi[pb], ai[pa], take_a);
    if (take_a) { pa+=1u; } else { pb+=1u; }
  }
  // Pack return in vec4; indices are staged to shared later
  return vec4<f32>(out[0], out[1], out[2], out[3]);
}

var<workgroup> svals: array<f32, 256u * 4u>;  // each thread writes 4; subgroup leaders compact later
var<workgroup> sidx : array<i32, 256u * 4u>;

@compute @workgroup_size(256)
fn main_cs(@builtin(local_invocation_id)  lid: vec3<u32>,
           @builtin(workgroup_id)        wid: vec3<u32>,
           @builtin(subgroup_size)       sg_size: u32,
           @builtin(subgroup_invocation_id) sg_lane: u32)
{
  let rows  = P.x;
  let total = P.y;
  let kfin  = P.z;
  let r = wid.x;
  if (r >= rows) { return; }

  // (1) Per‑thread scan → tiny keep‑4
  var kv: array<f32,4>;
  var ki: array<i32,4>;
  for (var j=0u;j<4u;j++){ kv[j] = -1e30; ki[j] = -1; }

  let tid = lid.x;
  var i = tid;
  loop {
    if (i >= total) { break; }
    let base = (r*total + i) * 2u;
    let p = unpack_at(base);
    let v  = f32_from_hi(p.y);
    let ix = i32_from_lo(p.x);
    insert4_desc(&kv, &ki, v, ix);
    i += 256u;
  }

  // (2) Subgroup reduction: pairwise merge keep‑4 to lane0
  var step = 1u;
  loop {
    if (step >= sg_size) { break; }
    if ((sg_lane % (2u*step)) == 0u) {
      let partner = sg_lane + step;
      if (partner < sg_size) {
        // pull partner kv/ki
        var bv: array<f32,4>;
        var bi: array<i32,4>;
        for (var j=0u;j<4u;j++){
          bv[j] = subgroupBroadcastFirst(kv[j], partner);
          bi[j] = subgroupBroadcastFirst(ki[j], partner);
        }
        // merge keep‑4
        // Re‑use insert4 for simplicity (4+4 merge)
        var tmpv: array<f32,4>;
        var tmpi: array<i32,4>;
        for (var j=0u;j<4u;j++){ tmpv[j] = -1e30; tmpi[j] = -1; }
        for (var j=0u;j<4u;j++){ insert4_desc(&tmpv, &tmpi, kv[j], ki[j]); }
        for (var j=0u;j<4u;j++){ insert4_desc(&tmpv, &tmpi, bv[j], bi[j]); }
        for (var j=0u;j<4u;j++){ kv[j]=tmpv[j]; ki[j]=tmpi[j]; }
      }
    }
    step = step << 1u;
  }

  // (3) Stage subgroup leaders to shared (lane0 of each subgroup)
  let wg_threads = 256u;
  let sgw = (wg_threads / sg_size);       // number of subgroups per WG
  if (sg_lane == 0u) {
    let sgid = (lid.x / sg_size);
    let base = (sgid * 4u);
    for (var j=0u;j<4u;j++){ svals[base+j] = kv[j]; sidx[base+j] = ki[j]; }
  }
  workgroupBarrier();

  // (4) Final selection on small pool: sgw * 4 items → pick first kfin
  if (lid.x == 0u) {
    let pool = sgw * 4u;
    let off  = r * kfin;
    // simple selection (kfin expected <= 1024 under this path)
    for (var j=0u; j<kfin; j++) {
      var best = j;
      for (var t=j+1u; t<pool; t++) {
        if (svals[t] > svals[best]) { best = t; }
      }
      // swap
      let tv = svals[best]; let ti = sidx[best];
      svals[best] = svals[j]; sidx[best] = sidx[j];
      svals[j] = tv;          sidx[j] = ti;
      out_vals.data[off+j] = svals[j];
      out_idx .data[off+j] = sidx[j];
    }
  }
}
