@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@group(0) @binding(3) var<storage, read_write> flags: array<atomic<u32>>;
var<workgroup> sums: array<f32,256>;

fn check(x:f32) {
    if ((bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&flags[0], INVALID_TENSOR_FLAG);
    }
}
// x-grid, groups, phase, source-offset, input-len, reduction-len, partials, rank,
// padded input shape, reduction shape, contiguous output strides.
fn source_index(element:u32, reduced:u32) -> u32 {
    var e = element;
    var r = reduced;
    let rank = params[7];
    var index = params[3];
    for (var axis=rank; axis>0u; axis--) {
        let d = axis-1u;
        let size = params[8u+d];
        let reduction_size = params[8u+rank+d];
        index += (e % size + r % reduction_size) * params[8u+2u*rank+d];
        e /= size;
        r /= reduction_size;
    }
    return index;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let group = wid.y*params[0]+wid.x;
    if (group >= params[1]) { return; }
    var value = 0.0;
    if (params[2] == 0u) {
        let element = group / params[6];
        let reduced = (group % params[6])*256u + lane;
        if (element < params[4] && reduced < params[5]) {
            value = source[source_index(element,reduced)];
            check(value);
        }
    } else {
        for (var chunk=lane; chunk<params[6]; chunk+=256u) {
            value += source[group*params[6]+chunk];
            check(value);
        }
    }
    sums[lane] = value;
    workgroupBarrier();
    for (var stride=128u; stride>0u; stride/=2u) {
        if (lane < stride) {
            sums[lane] += sums[lane+stride];
            check(sums[lane]);
        }
        workgroupBarrier();
    }
    if (lane == 0u && group < params[4]*select(params[6],1u,params[2]!=0u)) {
        out[group] = sums[0];
    }
}
