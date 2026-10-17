ROUNDED_ADD
WIDE_ARITHMETIC
ATTENTION_EXPONENTIAL

struct View {
    strides: vec4<u32>,
    offset: u32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
};
struct VjpParams {
    contexts: u32,
    queries: u32,
    keys: u32,
    head_dim: u32,
    scale: f32,
    flags: u32,
    query_offset: u32,
    heads: u32,
    query_groups_x: u32,
    key_groups_x: u32,
    stats_offset: u32,
    query_offset_out: u32,
    key_offset_out: u32,
    value_offset_out: u32,
    z_offset_out: u32,
    pair_offset_out: u32,
    query: View,
    key: View,
    value: View,
    z_bias: View,
    pair_bias: View,
    upstream: View,
};

@group(0) @binding(0) var<storage, read> queries: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> z_bias: array<f32>;
@group(0) @binding(4) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(5) var<storage, read> upstream: array<f32>;
@group(0) @binding(6) var<storage, read_write> output: array<f32>;
@group(0) @binding(7) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: VjpParams;

var<workgroup> dots: array<f32, 64>;
var<workgroup> dp_parts: array<WideSum, 64>;
var<workgroup> pair_score: f32;
var<workgroup> pair_dp: WideSum;
var<workgroup> row_maximum: f32;
var<workgroup> reference_key: u32;
var<workgroup> row_sum: WideSum;
var<workgroup> row_center: WideSum;
var<workgroup> probability: Wide;
var<workgroup> score_gradient: Wide;

fn checked(x: f32) -> f32 {
    if ((bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&flags[0], 0x80000000u);
        return 0.0;
    }
    return x;
}

fn base(view: View, context: u32, row: u32) -> u32 {
    return view.offset + (context / params.heads) * view.strides.x
        + (context % params.heads) * view.strides.y + row * view.strides.z;
}

fn visible_keys(query: u32) -> u32 {
    if ((params.flags & 4u) != 0u) { return params.query_offset + query + 1u; }
    return params.keys;
}

fn store_wide(offset: u32, value: Wide) {
    output[offset] = value.hi;
    output[offset + 1u] = value.lo;
    output[offset + 2u] = bitcast<f32>(value.exponent);
    output[offset + 3u] = value.tail;
}

fn load_wide(offset: u32) -> Wide {
    return Wide(output[offset], output[offset + 1u], bitcast<i32>(output[offset + 2u]), output[offset + 3u]);
}

// All callers enter with a uniform context/query/key and all 64 lanes.
fn compute_pair(context: u32, query: u32, key: u32, lane: u32, with_dp: bool, reference: u32) {
    let qb = base(params.query, context, query);
    let kb = base(params.key, context, key);
    let vb = base(params.value, context, key);
    let rb = base(params.value, context, reference);
    let ub = base(params.upstream, context, query);
    var dot = 0.0;
    var dp = wide_sum_zero();
    // Recompute the same score as forward, including its key-tile reduction.
    // A different tree can change cancellation by O(1), not merely one ULP.
    let dot_lanes = select(64u, 8u, (params.flags & 8u) != 0u);
    if (lane < dot_lanes) {
        for (var i = lane; i < params.head_dim; i += dot_lanes) {
            dot = checked(dot + checked(queries[qb + i * params.query.strides.w] * keys[kb + i * params.key.strides.w]));
        }
    }
    if (with_dp) {
        for (var i = lane; i < params.head_dim; i += 64u) {
            // Keep the shifted dot's tail through centering. Collapsing Vk-Vref
            // first can lose a small value against a huge reference value.
            let u = parts(upstream[ub + i * params.upstream.strides.w]);
            dp = wide_sum_add(dp, wide_mul(u, parts(values[vb + i * params.value.strides.w])));
            dp = wide_sum_add(dp, wide_neg(wide_mul(u, parts(values[rb + i * params.value.strides.w]))));
        }
    }
    dots[lane] = dot;
    dp_parts[lane] = dp;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            if (stride < dot_lanes) { dots[lane] = checked(dots[lane] + dots[lane + stride]); }
            if (with_dp) {
                dp_parts[lane] = wide_sum_add(wide_sum_add(dp_parts[lane], dp_parts[lane + stride].value), dp_parts[lane + stride].correction);
            }
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        var score = checked(dots[0] * params.scale);
        if ((params.flags & 1u) != 0u) {
            score = checked(score + z_bias[base(params.z_bias, context, 0u) + key * params.z_bias.strides.w]);
        }
        if ((params.flags & 2u) != 0u) {
            score = checked(score + pair_bias[base(params.pair_bias, context, query) + key * params.pair_bias.strides.w]);
        }
        pair_score = score;
        pair_dp = dp_parts[0];
    }
    workgroupBarrier();
}

@compute @workgroup_size(64)
fn statistics(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.y * params.query_groups_x + group.x;
    if (row >= params.contexts * params.queries) { return; }
    let context = row / params.queries;
    let query = row % params.queries;
    if (lane == 0u) {
        row_maximum = bitcast<f32>(0xff7fffffu);
        reference_key = 0u;
        row_sum = wide_sum_zero();
        row_center = wide_sum_zero();
    }
    workgroupBarrier();
    for (var key = 0u; key < visible_keys(query); key += 1u) {
        compute_pair(context, query, key, lane, false, 0u);
        if (lane == 0u && (key == 0u || pair_score > row_maximum)) {
            row_maximum = pair_score;
            reference_key = key;
        }
        workgroupBarrier();
    }
    // Derivatives recompute weights at this final maximum. Online rescaling
    // with rounded exponentials is not multiplicatively consistent with that
    // recomputation and can break both probability and score-gradient sums.
    for (var key = 0u; key < visible_keys(query); key += 1u) {
        compute_pair(context, query, key, lane, true, reference_key);
        if (lane == 0u) {
            let weight = score_weight(pair_score, row_maximum);
            row_sum = wide_sum_add(row_sum, weight);
            row_center = wide_sum_add(row_center, wide_mul(pair_dp.value, weight));
            row_center = wide_sum_add(row_center, wide_mul(pair_dp.correction, weight));
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        let offset = params.stats_offset + row * 14u;
        output[offset] = row_maximum;
        let denominator = wide_sum_value(row_sum);
        store_wide(offset + 1u, denominator);
        store_wide(offset + 5u, wide_div(row_center.value, denominator));
        store_wide(offset + 9u, wide_div(row_center.correction, denominator));
        output[offset + 13u] = bitcast<f32>(reference_key);
    }
}

fn compute_derivatives(row: u32, lane: u32) {
    if (lane == 0u) {
        let offset = params.stats_offset + row * 14u;
        probability = wide_div(score_weight(pair_score, output[offset]), load_wide(offset + 1u));
        let centered = wide_sum_add(wide_sum_add(pair_dp, wide_neg(load_wide(offset + 5u))), wide_neg(load_wide(offset + 9u)));
        score_gradient = wide_mul(probability, wide_sum_value(centered));
    }
    workgroupBarrier();
}

@compute @workgroup_size(64)
fn query_pullback(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.y * params.query_groups_x + group.x;
    if (row >= params.contexts * params.queries) { return; }
    let context = row / params.queries;
    let query = row % params.queries;
    var gradient: array<WideSum, 4>;
    for (var i = 0u; i < 4u; i += 1u) { gradient[i] = wide_sum_zero(); }
    let reference = bitcast<u32>(output[params.stats_offset + row * 14u + 13u]);
    for (var key = 0u; key < visible_keys(query); key += 1u) {
        compute_pair(context, query, key, lane, true, reference);
        compute_derivatives(row, lane);
        let scaled = wide_mul(parts(params.scale), score_gradient);
        let kb = base(params.key, context, key);
        for (var i = lane; i < params.head_dim; i += 64u) {
            gradient[i / 64u] = wide_sum_add(gradient[i / 64u], wide_mul(scaled, parts(keys[kb + i * params.key.strides.w])));
        }
        if (lane == 0u && (params.flags & 2u) != 0u) {
            output[params.pair_offset_out + row * params.keys + key] = checked(wide_float(score_gradient));
        }
        workgroupBarrier();
    }
    if ((params.flags & 2u) != 0u) {
        for (var key = visible_keys(query) + lane; key < params.keys; key += 64u) {
            output[params.pair_offset_out + row * params.keys + key] = 0.0;
        }
    }
    for (var i = lane; i < params.head_dim; i += 64u) {
        output[params.query_offset_out + row * params.head_dim + i] = checked(wide_float(wide_sum_value(gradient[i / 64u])));
    }
}

@compute @workgroup_size(64)
fn key_value_pullback(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.y * params.key_groups_x + group.x;
    if (row >= params.contexts * params.keys) { return; }
    let context = row / params.keys;
    let key = row % params.keys;
    var z_gradient = wide_sum_zero();
    var gradient_a: array<WideSum, 4>;
    var gradient_b: array<WideSum, 4>;
    for (var i = 0u; i < 4u; i += 1u) {
        gradient_a[i] = wide_sum_zero();
        gradient_b[i] = wide_sum_zero();
    }
    for (var query = 0u; query < params.queries; query += 1u) {
        if (key >= visible_keys(query)) { continue; }
        let reference = bitcast<u32>(output[params.stats_offset + (context * params.queries + query) * 14u + 13u]);
        compute_pair(context, query, key, lane, true, reference);
        compute_derivatives(context * params.queries + query, lane);
        let scaled = wide_mul(parts(params.scale), score_gradient);
        let qb = base(params.query, context, query);
        let ub = base(params.upstream, context, query);
        for (var i = lane; i < params.head_dim; i += 64u) {
            gradient_a[i / 64u] = wide_sum_add(gradient_a[i / 64u], wide_mul(scaled, parts(queries[qb + i * params.query.strides.w])));
            gradient_b[i / 64u] = wide_sum_add(gradient_b[i / 64u], wide_mul(probability, parts(upstream[ub + i * params.upstream.strides.w])));
        }
        if (lane == 0u) { z_gradient = wide_sum_add(z_gradient, score_gradient); }
        workgroupBarrier();
    }
    for (var i = lane; i < params.head_dim; i += 64u) {
        output[params.key_offset_out + row * params.head_dim + i] = checked(wide_float(wide_sum_value(gradient_a[i / 64u])));
        output[params.value_offset_out + row * params.head_dim + i] = checked(wide_float(wide_sum_value(gradient_b[i / 64u])));
    }
    if (lane == 0u && (params.flags & 1u) != 0u) {
        output[params.z_offset_out + row] = checked(wide_float(wide_sum_value(z_gradient)));
    }
}
