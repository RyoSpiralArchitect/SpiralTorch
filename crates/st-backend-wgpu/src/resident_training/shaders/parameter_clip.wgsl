struct ClipParams {
    len: u32,
    groups_x: u32,
    offset: u32,
    partials: u32,
    enabled: u32,
    limit_mantissa: f32,
    limit_exponent: i32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> inherited: array<u32>;
@group(0) @binding(2) var<storage, read_write> partials: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> factors: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@group(0) @binding(5) var<storage, read_write> validation: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> p: ClipParams;
var<workgroup> norm_pairs: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn clip_partials(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group = wid.x + p.groups_x * wid.y;
    let i = group * 256u + lane;
    var pair = vec2<f32>(0.0);
    if (i < p.len) {
        let bits = input[i];
        if ((bits & 0x7f800000u) == 0x7f800000u || inherited[0] != 0u) {
            atomicOr(&validation[0], INVALID_TENSOR_FLAG);
        } else if (p.enabled != 0u) {
            let magnitude = bits & 0x7fffffffu;
            if (magnitude != 0u) { pair = vec2<f32>(bitcast<f32>(magnitude), 1.0); }
        }
    }
    norm_pairs[lane] = pair;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) { norm_pairs[lane] = norm_merge(norm_pairs[lane], norm_pairs[lane + stride]); }
        workgroupBarrier();
    }
    if (lane == 0u && group < p.len / 256u + select(0u, 1u, p.len % 256u != 0u)) {
        partials[p.offset + group] = norm_pairs[0];
    }
}

@compute @workgroup_size(256)
fn clip_reduce(@builtin(local_invocation_index) lane: u32) {
    var pair = vec2<f32>(0.0);
    for (var i = lane; i < p.partials; i += 256u) { pair = norm_merge(pair, partials[i]); }
    norm_pairs[lane] = pair;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) { norm_pairs[lane] = norm_merge(norm_pairs[lane], norm_pairs[lane + stride]); }
        workgroupBarrier();
    }
    if (lane == 0u) {
        var clipped = global_clip_factors(norm_pairs[0], p.limit_mantissa, p.limit_exponent);
        if (clipped.invalid != 0u) { atomicOr(&validation[0], INVALID_TENSOR_FLAG); }
        factors[0] = f32(clipped.count);
        for (var i = 0u; i < clipped.count; i++) { factors[1u + i] = clipped.values[i]; }
    }
}

@compute @workgroup_size(256)
fn clip_apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = (wid.x + p.groups_x * wid.y) * 256u + lane;
    if (i >= p.len) { return; }
    let count = u32(factors[0]);
    if (p.enabled == 0u || count == 0u) { output[i] = input[i]; return; }
    var value = bitcast<f32>(input[i]);
    for (var n = 0u; n < count; n++) { value *= factors[1u + n]; }
    if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&validation[0], INVALID_TENSOR_FLAG);
    }
    output[i] = bitcast<u32>(value);
}
