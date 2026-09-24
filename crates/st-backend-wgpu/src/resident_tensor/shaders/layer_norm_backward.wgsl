ROUNDED_ADD
WIDE_ARITHMETIC

struct Params {
    rows: u32, cols: u32, groups_x: u32, requested: u32,
    epsilon: f32, scale: f32, beta_offset: u32, flag_slot: u32,
};
@group(0) @binding(0) var<storage, read> centered_values: array<Wide>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> upstream: array<f32>;
@group(0) @binding(3) var<storage, read> row_stats: array<LayerNormRow>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@group(0) @binding(5) var<storage, read_write> affine: array<f32>;
@group(0) @binding(6) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> sums: array<Wide, 256>;
var<workgroup> projections: array<Wide, 256>;
var<workgroup> mean: Wide;
var<workgroup> projection: Wide;
var<workgroup> epsilon_sum: Wide;
var<workgroup> input_denominator: Wide;
var<workgroup> combined_safe: bool;
var<workgroup> input_scale: Wide;
var<workgroup> fast_denominator: f32;
var<workgroup> fast_projection: f32;
var<workgroup> fast_scale: f32;
var<workgroup> fast_row_safe: bool;

fn checked(value: Wide) -> f32 {
    let result = wide_float(value);
    if ((bitcast<u32>(result) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&flags[params.flag_slot], 1u);
        return 0.0;
    }
    return result;
}

fn reduce(lane: u32) {
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            sums[lane] = wide_add(sums[lane], sums[lane + stride]);
            projections[lane] = wide_add(projections[lane], projections[lane + stride]);
        }
        workgroupBarrier();
    }
}

fn reduce_small(lane: u32, first_stride: u32) {
    workgroupBarrier();
    for (var stride = first_stride; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            sums[lane] = wide_add(sums[lane], sums[lane + stride]);
            projections[lane] = wide_add(projections[lane], projections[lane + stride]);
        }
        workgroupBarrier();
    }
}

fn weighted(index: u32, col: u32) -> Wide {
    return wide_mul(parts(upstream[index]), parts(gamma[col]));
}

@compute @workgroup_size(256)
fn backward_input(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.y * params.groups_x + group.x;
    if (row >= params.rows) { return; }
    let base = row * params.cols;
    let origin = weighted(base, 0u);
    var sum = parts(0.0);
    var dot = parts(0.0);
    for (var col = lane; col < params.cols; col += 256u) {
        let g = wide_sub(weighted(base + col, col), origin);
        sum = wide_add(sum, g);
        dot = wide_add(dot, wide_mul(g, centered_values[base + col]));
    }
    sums[lane] = sum;
    projections[lane] = dot;
    reduce(lane);
    if (lane == 0u) {
        mean = wide_div(sums[0], parts(f32(params.cols)));
        projection = projections[0];
        epsilon_sum = wide_mul(parts(params.epsilon), parts(f32(params.cols)));
        // This divisor is row-constant. Keep its extended-range reciprocal on
        // GPU instead of repeating compensated division for every input VJP.
        let stats = row_stats[row];
        input_denominator = wide_add(stats.square_sum, epsilon_sum);
        // Leave 20 exponent bits of margin: a distant epsilon can vanish from
        // the rounded row sum before a scale-direction cancellation.
        combined_safe = epsilon_sum.hi != 0.0 &&
                        (stats.square_sum.hi == 0.0 || epsilon_sum.exponent >= stats.square_sum.exponent - 20);
        input_scale = wide_div(stats.inverse_std, input_denominator);
        fast_denominator = wide_float(input_denominator);
        fast_projection = wide_float(projection);
        fast_scale = wide_float(input_scale);
        // Extreme row scales and weak epsilon stay on the Wide path.
        fast_row_safe = params.cols >= 32u && params.epsilon >= 1e-5 && combined_safe &&
                        fast_denominator > 1e-3 && fast_denominator < 1e8 &&
                        abs(fast_projection) < 1e8 &&
                        abs(fast_scale) > 1e-12 && abs(fast_scale) < 1e6;
    }
    workgroupBarrier();
    for (var col = lane; col < params.cols; col += 256u) {
        let g = wide_sub(weighted(base + col, col), origin);
        let stats = row_stats[row];
        // Cancel before dividing: squaring rounded normalized values introduces
        // a scale-direction residual that tiny variance can amplify enormously.
        let centered_g = wide_sub(g, mean);
        if (fast_row_safe) {
            let first = wide_float(centered_g) * fast_denominator;
            let second = wide_float(centered_values[base + col]) * fast_projection;
            let numerator = first - second;
            let magnitude = abs(first) + abs(second);
            // Cap cancellation amplification of f32 rounding at 128x.
            if (magnitude > 0.0 && magnitude < 1e20 &&
                abs(numerator) >= magnitude / 128.0) {
                let value = numerator * fast_scale;
                if (abs(value) < 1e30) {
                    dx[base + col] = value;
                    continue;
                }
            }
        }
        var numerator: Wide;
        if (combined_safe) {
            numerator = wide_difference_of_products(centered_g, input_denominator,
                                                     centered_values[base + col], projection);
        } else {
            let variance_residual = wide_difference_of_products(centered_g, stats.square_sum,
                                                                centered_values[base + col], projection);
            numerator = wide_add(variance_residual, wide_mul(centered_g, epsilon_sum));
        }
        dx[base + col] = checked(wide_mul(numerator, input_scale));
    }
}

// Keep the 256-lane entry point below unchanged as the broad-shape reference.
fn compute_affine_small(group: vec3<u32>, lane: u32, lanes: u32) {
    let col = group.y * params.groups_x + group.x;
    if (col >= params.cols) { return; }
    var dg = parts(0.0);
    var db = parts(0.0);
    for (var row = lane; row < params.rows; row += lanes) {
        let index = row * params.cols + col;
        let seed = parts(upstream[index]);
        if ((params.requested & 2u) != 0u) {
            let normalized = wide_mul(centered_values[index], row_stats[row].inverse_std);
            dg = wide_add(dg, wide_mul(seed, normalized));
        }
        if ((params.requested & 4u) != 0u) { db = wide_add(db, seed); }
    }
    sums[lane] = dg;
    projections[lane] = db;
    reduce_small(lane, lanes / 2u);
    if (lane == 0u) {
        let scale = parts(params.scale);
        if ((params.requested & 2u) != 0u) { affine[col] = checked(wide_mul(sums[0], scale)); }
        if ((params.requested & 4u) != 0u) { affine[params.beta_offset + col] = checked(wide_mul(projections[0], scale)); }
    }
}

@compute @workgroup_size(64)
fn backward_affine_64(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    compute_affine_small(group, lane, 64u);
}

@compute @workgroup_size(128)
fn backward_affine_128(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    compute_affine_small(group, lane, 128u);
}

@compute @workgroup_size(256)
fn backward_affine(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let col = group.y * params.groups_x + group.x;
    if (col >= params.cols) { return; }
    var dg = parts(0.0);
    var db = parts(0.0);
    for (var row = lane; row < params.rows; row += 256u) {
        let index = row * params.cols + col;
        let seed = parts(upstream[index]);
        if ((params.requested & 2u) != 0u) {
            let normalized = wide_mul(centered_values[index], row_stats[row].inverse_std);
            dg = wide_add(dg, wide_mul(seed, normalized));
        }
        if ((params.requested & 4u) != 0u) { db = wide_add(db, seed); }
    }
    sums[lane] = dg;
    projections[lane] = db;
    reduce(lane);
    if (lane == 0u) {
        let scale = parts(params.scale);
        if ((params.requested & 2u) != 0u) { affine[col] = checked(wide_mul(sums[0], scale)); }
        if ((params.requested & 4u) != 0u) { affine[params.beta_offset + col] = checked(wide_mul(projections[0], scale)); }
    }
}

// Eight adjacent columns share a workgroup. Each column still has an ordered
// 32-lane row reduction, while adjacent lanes read adjacent addresses.
@compute @workgroup_size(256)
fn backward_affine_tiled(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let col_lane = lane & 7u;
    let row_lane = lane >> 3u;
    let tile = group.y * params.groups_x + group.x;
    let col = tile * 8u + col_lane;
    var dg = parts(0.0);
    var db = parts(0.0);
    if (col < params.cols) {
        for (var row = row_lane; row < params.rows; row += 32u) {
            let index = row * params.cols + col;
            let seed = parts(upstream[index]);
            if ((params.requested & 2u) != 0u) {
                let normalized = wide_mul(centered_values[index], row_stats[row].inverse_std);
                dg = wide_add(dg, wide_mul(seed, normalized));
            }
            if ((params.requested & 4u) != 0u) { db = wide_add(db, seed); }
        }
    }
    sums[lane] = dg;
    projections[lane] = db;
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride >>= 1u) {
        if (row_lane < stride) {
            let other = lane + stride * 8u;
            sums[lane] = wide_add(sums[lane], sums[other]);
            projections[lane] = wide_add(projections[lane], projections[other]);
        }
        workgroupBarrier();
    }
    if (row_lane == 0u && col < params.cols) {
        let scale = parts(params.scale);
        if ((params.requested & 2u) != 0u) { affine[col] = checked(wide_mul(sums[lane], scale)); }
        if ((params.requested & 4u) != 0u) { affine[params.beta_offset + col] = checked(wide_mul(projections[lane], scale)); }
    }
}
