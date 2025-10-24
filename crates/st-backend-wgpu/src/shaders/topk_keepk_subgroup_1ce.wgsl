enable chromium_experimental_subgroups;

struct InP  { data: array<u32>, }; // u64 packed [lo(idx), hi(f32_bits)]
struct OutV { data: array<f32>, };
struct OutI { data: array<i32>, };

// rows, total(cols), k_final, reserved
@group(0) @binding(3) var<uniform> P : vec4<u32>;

@group(0) @binding(0) var<storage, read>        packed  : InP;
@group(0) @binding(1) var<storage, read_write>  out_vals: OutV;
@group(0) @binding(2) var<storage, read_write>  out_idx : OutI;

fn unpack2(base:u32)->vec2<u32>{
  let lo = packed.data[base];
  let hi = packed.data[base+1];
  return vec2<u32>(lo, hi);
}
fn f32_from_hi(hi:u32)->f32 { return bitcast<f32>(hi); }
fn i32_from_lo(lo:u32)->i32 { return bitcast<i32>(lo); }

fn insert4_desc(mut kv: ptr<function, array<f32,4>>,
                mut ki: ptr<function, array<i32,4>>,
                v:f32, ix:i32){
  if (v <= (*kv)[3u]) { return; }
  (*kv)[3u] = v; (*ki)[3u] = ix;
  if ((*kv)[3u] > (*kv)[2u]) { let tv=(*kv)[2u]; let ti=(*ki)[2u]; (*kv)[2u]=(*kv)[3u]; (*ki)[2u]=(*ki)[3u]; (*kv)[3u]=tv; (*ki)[3u]=ti; }
  if ((*kv)[2u] > (*kv)[1u]) { let tv=(*kv)[1u]; let ti=(*ki)[1u]; (*kv)[1u]=(*kv)[2u]; (*ki)[1u]=(*ki)[2u]; (*kv)[2u]=tv; (*ki)[2u]=ti; }
  if ((*kv)[1u] > (*kv)[0u]) { let tv=(*kv)[0u]; let ti=(*ki)[0u]; (*kv)[0u]=(*kv)[1u]; (*ki)[0u]=(*ki)[1u]; (*kv)[1u]=tv; (*ki)[1u]=ti; }
}

var<workgroup> svals: array<f32, 256u * 4u>;
var<workgroup> sidx : array<i32, 256u * 4u>;

@compute @workgroup_size(256)
fn main_cs(@builtin(local_invocation_id)  lid: vec3<u32>,
           @builtin(workgroup_id)        wid: vec3<u32>,
           @builtin(subgroup_size)       sg_size: u32,
           @builtin(subgroup_invocation_id) sg_lane: u32)
{
  let rows  = P.x;
  let cols  = P.y;
  let kfin  = P.z;
  let r = wid.x;
  if (r >= rows) { return; }

  // Per-thread tiny keep-4 over the row (stride=WG size)
  var kv: array<f32,4>; var ki: array<i32,4>;
  for (var j=0u;j<4u;j++){ kv[j] = -1e30; ki[j] = -1; }

  let tid = lid.x;
  var c = tid;
  loop {
    if (c >= cols) { break; }
    let base = (r*cols + c) * 2u;
    let p = unpack2(base);
    insert4_desc(&kv, &ki, f32_from_hi(p.y), i32_from_lo(p.x));
    c += 256u;
  }

  // In-subgroup reduction (pairwise; lane0 keeps top-4)
  var step = 1u;
  loop {
    if (step >= sg_size) { break; }
    if ((sg_lane % (2u*step)) == 0u) {
      let partner = sg_lane + step;
      if (partner < sg_size) {
        var bv: array<f32,4>; var bi: array<i32,4>;
        for (var j=0u;j<4u;j++){
          bv[j] = subgroupBroadcastFirst(kv[j], partner);
          bi[j] = subgroupBroadcastFirst(ki[j], partner);
        }
        var tmpv: array<f32,4>; var tmpi: array<i32,4>;
        for (var j=0u;j<4u;j++){ tmpv[j]=-1e30; tmpi[j]=-1; }
        for (var j=0u;j<4u;j++){ insert4_desc(&tmpv, &tmpi, kv[j], ki[j]); }
        for (var j=0u;j<4u;j++){ insert4_desc(&tmpv, &tmpi, bv[j], bi[j]); }
        for (var j=0u;j<4u;j++){ kv[j]=tmpv[j]; ki[j]=tmpi[j]; }
      }
    }
    step = step << 1u;
  }

  // Subgroup leaders write to shared. Auto-control pool size:
  // keep_m = clamp(ceil(kfin / num_subgroups), 1..4)
  let sgw = (256u / sg_size); // number of subgroups in WG
  var keep_m = (kfin + sgw - 1u) / sgw;
  if (keep_m < 1u) { keep_m = 1u; }
  if (keep_m > 4u) { keep_m = 4u; }

  if (sg_lane == 0u) {
    let sgid = (lid.x / sg_size);
    let base = sgid * 4u;
    for (var j=0u; j<keep_m; j++){
      svals[base+j] = kv[j];
      sidx [base+j] = ki[j];
    }
  }
  workgroupBarrier();

  // Final selection on small pool: pool = sgw * keep_m
  if (lid.x == 0u) {
    let pool = sgw * keep_m;
    let off  = r * kfin;
    // (selection sort; pool is small)
    var j=0u;
    loop {
      if (j >= kfin) { break; }
      var best = j;
      var t = j+1u;
      loop {
        if (t >= pool) { break; }
        if (svals[t] > svals[best]) { best = t; }
        t += 1u;
      }
      let tv = svals[best]; let ti = sidx[best];
      svals[best] = svals[j]; sidx[best] = sidx[j];
      svals[j] = tv;         sidx[j] = ti;
      out_vals.data[off+j] = svals[j];
      out_idx .data[off+j] = sidx[j];
      j += 1u;
    }
  }
}
