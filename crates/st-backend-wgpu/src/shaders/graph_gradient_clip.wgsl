// Appended to dense_training.wgsl. Global norm of policy-normalized gradients.
// A pair is (largest magnitude, sum of squared values divided by that magnitude).
var<workgroup> norm_pairs: array<vec2<f32>, 256>;

fn norm_merge(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    if (a.x == 0.0) { return b; }
    if (b.x == 0.0) { return a; }
    let hi = max(a.x, b.x);
    // A direct division by a wide magnitude may lower to a flushed reciprocal.
    let h = frexp(hi);
    let x = frexp(a.x);
    let y = frexp(b.x);
    let ra = ldexp(x.fract / h.fract, x.exp - h.exp);
    let rb = ldexp(y.fract / h.fract, y.exp - h.exp);
    return vec2<f32>(hi, a.y * ra * ra + b.y * rb * rb);
}

fn effective(value: f32) -> f32 {
    check(value, 4096u);
    var scale = 1.0;
    if (p.gelu != 0u && step._pad0 != 0.0) { scale = 1.0 / f32(p.rows); }
    let result = value * scale;
    check(result, 4096u);
    return result;
}

@compute @workgroup_size(256)
fn clip_partials(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    var value = vec2<f32>(0.0);
    if (i < p.len) {
        let g = effective(a[i]);
        if ((bitcast<u32>(g) & 0x7f800000u) != 0x7f800000u && g != 0.0) {
            value = vec2<f32>(abs(g), 1.0);
        }
    }
    norm_pairs[lane] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) { norm_pairs[lane] = norm_merge(norm_pairs[lane], norm_pairs[lane + stride]); }
        workgroupBarrier();
    }
    let group = group_index(wid);
    if (lane == 0u && group < p.partials) {
        let offset = 2u * (p.cols + group);
        out[offset] = norm_pairs[0].x;
        out[offset + 1u] = norm_pairs[0].y;
    }
}

@compute @workgroup_size(256)
fn clip_reduce(@builtin(local_invocation_index) lane: u32) {
    var value = vec2<f32>(0.0);
    for (var i = lane; i < p.partials; i += 256u) {
        value = norm_merge(value, vec2<f32>(a[2u * i], a[2u * i + 1u]));
    }
    norm_pairs[lane] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) { norm_pairs[lane] = norm_merge(norm_pairs[lane], norm_pairs[lane + stride]); }
        workgroupBarrier();
    }
    if (lane == 0u) {
        out[0] = 0.0;
        let maximum = norm_pairs[0].x;
        let root = sqrt(norm_pairs[0].y);
        check(maximum, 4096u); check(root, 4096u);
        if (maximum == 0.0 || maximum <= CLIP_NORM_FLOOR / root) { return; }
        // Never materialize a possibly overflowing norm or underflowing scale.
        let denominator = frexp(maximum);
        let quotient = frexp((step._pad1 / denominator.fract) / root);
        var exponent = i32(step._pad2) - denominator.exp + quotient.exp;
        let mantissa = quotient.fract;
        if (exponent >= 1) { return; }
        var count = 0u;
        while (exponent < -64) {
            out[1u + count] = CLIP_SCALE_CHUNK;
            count += 1u;
            exponent += 64;
        }
        let last = ldexp(mantissa, exponent);
        if (count == 0u && abs(last - 1.0) <= CLIP_SCALE_EPSILON) { return; }
        check(last, 4096u);
        out[1u + count] = last;
        out[0] = f32(count + 1u);
    }
}

@compute @workgroup_size(256)
fn clip_prepare(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i < p.len) {
        var gradient = effective(c[i]);
        let count = u32(b[0]);
        for (var j = 0u; j < count; j += 1u) { gradient *= b[1u + j]; }
        let change = step.rate * gradient;
        let candidate = a[i] - change;
        check(gradient, 4096u); check(change, 8192u); check(candidate, 16384u);
        out[i] = candidate;
        aux[i] = gradient;
    }
}
