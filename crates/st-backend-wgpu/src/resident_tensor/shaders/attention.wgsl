struct Params {
    contexts: u32,
    queries: u32,
    keys: u32,
    head_dim: u32,
    scale: f32,
    flags: u32,
    query_offset: u32,
    groups_x: u32,
};

@group(0) @binding(0) var<storage, read> queries: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> z_bias: array<f32>;
@group(0) @binding(4) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> shared_q: array<f32, 256>;
var<workgroup> accum: array<f32, 256>;
var<workgroup> partials: array<f32, 64>;
var<workgroup> running_max: f32;
var<workgroup> running_sum: f32;
var<workgroup> alpha: f32;
var<workgroup> weight: f32;

fn checked(x: f32) -> f32 {
    if ((bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&flags[0], 0x80000000u);
        return 0.0;
    }
    return x;
}

@compute @workgroup_size(64)
fn forward(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.y * params.groups_x + group.x;
    if (row >= params.contexts * params.queries) { return; }
    let context = row / params.queries;
    let query = row % params.queries;
    let d = params.head_dim;
    for (var i = lane; i < d; i += 64u) {
        shared_q[i] = queries[row * d + i];
        accum[i] = 0.0;
    }
    if (lane == 0u) {
        running_max = bitcast<f32>(0xff7fffffu);
        running_sum = 0.0;
    }
    workgroupBarrier();
    var visible = params.keys;
    if ((params.flags & 4u) != 0u) { visible = params.query_offset + query + 1u; }
    for (var k = 0u; k < visible; k += 1u) {
        let key_row = context * params.keys + k;
        var dot = 0.0;
        for (var i = lane; i < d; i += 64u) {
            dot = checked(dot + checked(shared_q[i] * keys[key_row * d + i]));
        }
        partials[lane] = dot;
        workgroupBarrier();
        for (var stride = 32u; stride > 0u; stride >>= 1u) {
            if (lane < stride) { partials[lane] = checked(partials[lane] + partials[lane + stride]); }
            workgroupBarrier();
        }
        if (lane == 0u) {
            var score = checked(partials[0] * params.scale);
            if ((params.flags & 1u) != 0u) { score = checked(score + z_bias[key_row]); }
            if ((params.flags & 2u) != 0u) { score = checked(score + pair_bias[row * params.keys + k]); }
            let next_max = max(running_max, score);
            let previous = running_sum * exp(running_max - next_max);
            let current = exp(score - next_max);
            let sum = checked(previous + current);
            alpha = checked(previous / sum);
            weight = checked(current / sum);
            running_sum = sum;
            running_max = next_max;
        }
        workgroupBarrier();
        for (var i = lane; i < d; i += 64u) {
            accum[i] = checked(accum[i] * alpha + values[key_row * d + i] * weight);
        }
        workgroupBarrier();
    }
    for (var i = lane; i < d; i += 64u) { output[row * d + i] = accum[i]; }
}
