ROUNDED_ADD
WIDE_ARITHMETIC

struct Params {
    rows: u32, cols: u32, groups_x: u32, requested: u32,
    epsilon: f32, scale: f32, beta_offset: u32, _pad: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> centered_values: array<Wide>;
@group(0) @binding(4) var<storage, read_write> row_stats: array<LayerNormRow>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;

var<workgroup> sums: array<Wide, 256>;
var<workgroup> row_mean: Wide;
var<workgroup> row_inverse: Wide;

fn checked(value: f32) -> f32 {
    if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&flags[0], 1u);
        return 0.0;
    }
    return value;
}

fn reduce(lane: u32) {
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lane < stride) { sums[lane] = wide_add(sums[lane], sums[lane + stride]); }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn forward(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.y * params.groups_x + group.x;
    if (row >= params.rows) { return; }
    let base = row * params.cols;
    let origin = parts(input[base]);
    var sum = parts(0.0);
    for (var col = lane; col < params.cols; col += 256u) {
        sum = wide_add(sum, wide_sub(parts(input[base + col]), origin));
    }
    sums[lane] = sum;
    reduce(lane);
    if (lane == 0u) { row_mean = wide_div(sums[0], parts(f32(params.cols))); }
    workgroupBarrier();
    sum = parts(0.0);
    for (var col = lane; col < params.cols; col += 256u) {
        let centered = wide_sub(wide_sub(parts(input[base + col]), origin), row_mean);
        sum = wide_add(sum, wide_mul(centered, centered));
    }
    // All lanes consumed the first reduction before reusing its scratch.
    workgroupBarrier();
    sums[lane] = sum;
    reduce(lane);
    if (lane == 0u) {
        let square_sum = wide_add(sums[0], wide_mul(parts(params.epsilon), parts(f32(params.cols))));
        let denominator = wide_sqrt(wide_div(square_sum, parts(f32(params.cols))));
        if (denominator.hi == 0.0) { atomicOr(&flags[0], 1u); }
        row_inverse = wide_div(parts(1.0), denominator);
        // Retain the variance separately so epsilon survives cancellation in dx.
        row_stats[row] = LayerNormRow(row_inverse, sums[0]);
    }
    workgroupBarrier();
    for (var col = lane; col < params.cols; col += 256u) {
        let centered = wide_sub(wide_sub(parts(input[base + col]), origin), row_mean);
        let normed = wide_mul(centered, row_inverse);
        centered_values[base + col] = centered;
        // Forward retains the existing f32 affine contract. Backward consumes
        // the unrounded, extended-range centered tape instead.
        // Decode the rounded f32 bits again: a subnormal normalized value can
        // have a normal affine product, but hardware multiplication may flush
        // that operand to zero. Preserve the CPU's two f32 rounding boundaries.
        let product = wide_float(wide_mul(parts(wide_float(normed)), parts(gamma[col])));
        output[base + col] = checked(rounded_add(product, beta[col]));
    }
}
