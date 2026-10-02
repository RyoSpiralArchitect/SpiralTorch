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
override KEY_TILE: u32 = 1u;

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
var<workgroup> alpha: array<f32, 4>;
var<workgroup> weight: array<f32, 4>;

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
    // Specialized to one 64-lane or four 16-lane dot products.
    // The online normalization and value accumulation retain key order.
    let dot_lanes = 64u / KEY_TILE;
    let key_slot = lane / dot_lanes;
    let dot_lane = lane % dot_lanes;
    for (var first = 0u; first < visible; first += KEY_TILE) {
        let k = first + key_slot;
        let key_row = context * params.keys + k;
        var dot = 0.0;
        if (k < visible) {
            for (var i = dot_lane; i < d; i += dot_lanes) {
                dot = checked(dot + checked(shared_q[i] * keys[key_row * d + i]));
            }
        }
        partials[lane] = dot;
        workgroupBarrier();
        for (var stride = dot_lanes / 2u; stride > 0u; stride >>= 1u) {
            if (dot_lane < stride) { partials[lane] = checked(partials[lane] + partials[lane + stride]); }
            workgroupBarrier();
        }
        if (lane == 0u) {
            for (var slot = 0u; slot < min(KEY_TILE, visible - first); slot += 1u) {
                let key_index = first + slot;
                var score = checked(partials[slot * dot_lanes] * params.scale);
                if ((params.flags & 1u) != 0u) { score = checked(score + z_bias[context * params.keys + key_index]); }
                if ((params.flags & 2u) != 0u) { score = checked(score + pair_bias[row * params.keys + key_index]); }
                let next_max = max(running_max, score);
                let previous = running_sum * exp(running_max - next_max);
                let current = exp(score - next_max);
                let sum = checked(previous + current);
                alpha[slot] = checked(previous / sum);
                weight[slot] = checked(current / sum);
                running_sum = sum;
                running_max = next_max;
            }
        }
        workgroupBarrier();
        for (var i = lane; i < d; i += 64u) {
            for (var slot = 0u; slot < min(KEY_TILE, visible - first); slot += 1u) {
                let value_row = context * params.keys + first + slot;
                accum[i] = checked(accum[i] * alpha[slot] + values[value_row * d + i] * weight[slot]);
            }
        }
        workgroupBarrier();
    }
    for (var i = lane; i < d; i += 64u) { output[row * d + i] = accum[i]; }
}
