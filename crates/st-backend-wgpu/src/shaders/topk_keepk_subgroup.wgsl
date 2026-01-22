struct InBuf { data: array<u32>, };    // packed u64 as 2x u32
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
fn f32_from_hi(hi: u32) -> f32 {
    return bitcast<f32>(hi);
}
fn i32_from_lo(lo: u32) -> i32 {
    return bitcast<i32>(lo);
}

@compute @workgroup_size(256)
fn main_cs(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(local_invocation_id)  lid: vec3<u32>,
           @builtin(num_workgroups) num_wg: vec3<u32>,
           @builtin(workgroup_id)   wid: vec3<u32>,
           @builtin(subgroup_size) sg_size: u32,
           @builtin(subgroup_invocation_id) sg_lane: u32)
{
    let rows    = params.x;
    let total   = params.y;
    let k_final = params.z;

    let r = wid.x;
    if (r >= rows) { return; }

    // per-lane tiny keep-k (M=4)
    var kv: array<f32, 4>;
    var ki: array<i32, 4>;
    for (var j=0u; j<4u; j++){ kv[j] = -1e30; ki[j] = -1; }

    // strided scan
    let block = 256u;
    let tid = lid.x;
    var i = tid;
    loop {
        if (i >= total) { break; }
        let base = (r*total + i) * 2u;
        let p = unpack_at(base);
        let v = f32_from_hi(p.y);
        let ix = i32_from_lo(p.x);
        // insert-desc 4
        if (v > kv[3u]) {
            kv[3u] = v; ki[3u] = ix;
            if (kv[3u] > kv[2u]) { let tv=kv[2u]; let ti=ki[2u]; kv[2u]=kv[3u]; ki[2u]=ki[3u]; kv[3u]=tv; ki[3u]=ti; }
            if (kv[2u] > kv[1u]) { let tv=kv[1u]; let ti=ki[1u]; kv[1u]=kv[2u]; ki[1u]=ki[2u]; kv[2u]=tv; ki[2u]=ti; }
            if (kv[1u] > kv[0u]) { let tv=kv[0u]; let ti=ki[0u]; kv[0u]=kv[1u]; ki[0u]=ki[1u]; kv[1u]=tv; ki[1u]=ti; }
        }
        i += block;
    }

    // subgroup pairwise merge to lane0
    var lane = sg_lane;
    var step = 1u;
    loop {
        if (step >= sg_size) { break; }
        if ((lane % (2u*step)) == 0u) {
            // pull partner list via subgroupBroadcast
            let partner = lane + step;
            var bv: array<f32,4>;
            var bi: array<i32,4>;
            for (var j=0u; j<4u; j++){
                bv[j] = subgroupBroadcastFirst(kv[j], partner);
                bi[j] = subgroupBroadcastFirst(ki[j], partner);
            }
            // merge keep top-4
            var pa=0u; var pb=0u;
            var outv: array<f32,4>;
            var outi: array<i32,4>;
            for (var j=0u; j<4u; j++){
                let va = select(-1e30, kv[pa], pa<4u);
                let vb = select(-1e30, bv[pb], pb<4u);
                let take_a = va >= vb;
                outv[j] = select(vb, va, take_a);
                outi[j] = select(bi[pb], ki[pa], take_a);
                if (take_a) { pa+=1u; } else { pb+=1u; }
            }
            for (var j=0u; j<4u; j++){ kv[j]=outv[j]; ki[j]=outi[j]; }
        }
        step = step << 1u;
    }

    // lane0 writes
    if (sg_lane == 0u && lid.x < 256u) {
        // small selection over W*(4) stored into out (single row only demo)
        let base = r*k_final;
        // naive select (k <= 1024 assumed)
        // NOTE: for brevity we just write top-4 of this subgroup into the first slots
        for (var j=0u; j<min(4u, k_final); j++){
            out_vals.data[base+j] = kv[j];
            out_idx.data[base+j]  = ki[j];
        }
    }
}
