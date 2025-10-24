struct InBuf { data: array<u32>, };
struct OutVal { data: array<f32>, };
struct OutIdx { data: array<i32>, };

@group(0) @binding(0) var<storage, read>  packed: InBuf;
@group(0) @binding(1) var<storage, read_write> out_vals: OutVal;
@group(0) @binding(2) var<storage, read_write> out_idx: OutIdx;

@group(0) @binding(3) var<uniform> params: vec4<u32>; // rows, total, k_final, stride

fn unpack_at(base:u32)->vec2<u32>{
    let lo = packed.data[base];
    let hi = packed.data[base+1];
    return vec2<u32>(lo, hi);
}
fn f32_from_hi(hi: u32) -> f32 { return bitcast<f32>(hi); }
fn i32_from_lo(lo: u32) -> i32 { return bitcast<i32>(lo); }

var<workgroup> svals: array<f32, 256*4>;
var<workgroup> sidx:  array<i32, 256*4>;

@compute @workgroup_size(256)
fn main_cs(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(local_invocation_id)  lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>)
{
    let rows    = params.x;
    let total   = params.y;
    let k_final = params.z;
    let r = wid.x;
    if (r >= rows) { return; }
    let tid = lid.x;

    // per-thread tiny keep-k (4)
    var kv: array<f32, 4>;
    var ki: array<i32, 4>;
    for (var j=0u; j<4u; j++){ kv[j] = -1e30; ki[j] = -1; }
    var i = tid;
    loop {
        if (i >= total) { break; }
        let base = (r*total + i) * 2u;
        let p = unpack_at(base);
        let v = f32_from_hi(p.y);
        let ix = i32_from_lo(p.x);
        if (v > kv[3u]) {
            kv[3u] = v; ki[3u] = ix;
            if (kv[3u] > kv[2u]) { let tv=kv[2u]; let ti=ki[2u]; kv[2u]=kv[3u]; ki[2u]=ki[3u]; kv[3u]=tv; ki[3u]=ti; }
            if (kv[2u] > kv[1u]) { let tv=kv[1u]; let ti=ki[1u]; kv[1u]=kv[2u]; ki[1u]=ki[2u]; kv[2u]=tv; ki[2u]=ti; }
            if (kv[1u] > kv[0u]) { let tv=kv[0u]; let ti=ki[0u]; kv[0u]=kv[1u]; ki[0u]=ki[1u]; kv[1u]=tv; ki[1u]=ti; }
        }
        i += 256u;
    }

    // stage to shared
    let base = tid*4u;
    for (var j=0u; j<4u; j++){ svals[base+j]=kv[j]; sidx[base+j]=ki[j]; }
    workgroupBarrier();

    // simple selection (k small): block pool = 256*4
    if (tid==0u){
        let pool = 256u*4u;
        let off = r*k_final;
        for (var j=0u; j<k_final; j++){
            var best = j;
            for (var t=j+1u; t<pool; t++){
                if (svals[t] > svals[best]) { best = t; }
            }
            let tv = svals[best]; let ti = sidx[best];
            svals[best] = svals[j]; sidx[best] = sidx[j];
            svals[j] = tv; sidx[j] = ti;
            out_vals.data[off+j] = svals[j];
            out_idx .data[off+j] = sidx[j];
        }
    }
}
