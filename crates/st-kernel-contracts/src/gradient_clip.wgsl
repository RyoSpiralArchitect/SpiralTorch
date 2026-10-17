struct ClipBinaryParts { fract: f32, exp: i32 }

// Unlike frexp, this decomposition also has defined subnormal inputs.
fn clip_positive_parts(value: f32) -> ClipBinaryParts {
    let bits = bitcast<u32>(value);
    let exponent = bits >> 23u;
    if (exponent == 0u) {
        if (bits == 0u) { return ClipBinaryParts(0.0, 0); }
        let shift = countLeadingZeros(bits) - 8u;
        return ClipBinaryParts(bitcast<f32>(((bits << shift) & 0x7fffffu) | 0x3f000000u), -125 - i32(shift));
    }
    return ClipBinaryParts(bitcast<f32>((bits & 0x7fffffu) | 0x3f000000u), i32(exponent) - 126);
}

// Exact unsigned 32x32 product, least-significant word first.
fn clip_mul_words(a: u32, b: u32) -> vec2<u32> {
    let p0 = (a & 0xffffu) * (b & 0xffffu);
    let p1 = (a >> 16u) * (b & 0xffffu);
    let p2 = (a & 0xffffu) * (b >> 16u);
    let middle = (p0 >> 16u) + (p1 & 0xffffu) + (p2 & 0xffffu);
    return vec2<u32>((p0 & 0xffffu) | (middle << 16u),
        (a >> 16u) * (b >> 16u) + (p1 >> 16u) + (p2 >> 16u) + (middle >> 16u));
}

// Compare the represented maximum^2 * normalized sum to the power-of-two floor
// without a rounded division, sqrt or product at that discontinuous boundary.
// Norm merging itself remains f32, not exact real-valued norm accumulation.
fn clip_norm_exceeds_floor(pair: vec2<f32>) -> bool {
    let x = clip_positive_parts(pair.x);
    let y = clip_positive_parts(pair.y);
    let mx = (bitcast<u32>(x.fract) & 0x7fffffu) | 0x800000u;
    let my = (bitcast<u32>(y.fract) & 0x7fffffu) | 0x800000u;
    let square = clip_mul_words(mx, mx);
    let low = clip_mul_words(square.x, my);
    let high = clip_mul_words(square.y, my);
    let middle = low.y + high.x;
    let product = vec3<u32>(low.x, middle, high.y + select(0u, 1u, middle < low.y));
    let floor_exponent = clip_positive_parts(CLIP_NORM_FLOOR).exp - 1;
    let bit = 2 * floor_exponent - 2 * x.exp - y.exp + 72;
    if (bit < 0) { return true; }
    if (bit >= 96) { return false; }
    var threshold = vec3<u32>(0u);
    threshold[u32(bit) / 32u] = 1u << (u32(bit) % 32u);
    if (product.z != threshold.z) { return product.z > threshold.z; }
    if (product.y != threshold.y) { return product.y > threshold.y; }
    return product.x > threshold.x;
}

// A pair retains maximum magnitude and the sum of squares relative to it.
fn norm_merge(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    if (bitcast<u32>(a.x) == 0u) { return b; }
    if (bitcast<u32>(b.x) == 0u) { return a; }
    let hi = bitcast<f32>(max(bitcast<u32>(a.x), bitcast<u32>(b.x)));
    let h = clip_positive_parts(hi);
    let x = clip_positive_parts(a.x);
    let y = clip_positive_parts(b.x);
    let ra = ldexp(x.fract / h.fract, x.exp - h.exp);
    let rb = ldexp(y.fract / h.fract, y.exp - h.exp);
    return vec2<f32>(hi, a.y * ra * ra + b.y * rb * rb);
}

struct GlobalClipFactors {
    count: u32,
    invalid: u32,
    values: array<f32, CLIP_MAX_FACTORS>,
}

fn global_clip_factors(pair: vec2<f32>, limit_mantissa: f32, limit_exponent: i32) -> GlobalClipFactors {
    var result: GlobalClipFactors;
    let maximum = pair.x;
    let root = sqrt(pair.y);
    if ((bitcast<u32>(maximum) & 0x7f800000u) == 0x7f800000u ||
        (bitcast<u32>(root) & 0x7f800000u) == 0x7f800000u || maximum < 0.0 || pair.y < 0.0) {
        result.invalid = 1u;
        return result;
    }
    if (bitcast<u32>(maximum) == 0u) { return result; }
    if (root == 0.0) { result.invalid = 1u; return result; }
    if (!clip_norm_exceeds_floor(pair)) { return result; }
    // Neither the full norm nor a possibly subnormal scale is materialized.
    let denominator = clip_positive_parts(maximum);
    let quotient = frexp((limit_mantissa / denominator.fract) / root);
    var exponent = limit_exponent - denominator.exp + quotient.exp;
    if (exponent >= 1) { return result; }
    while (exponent < -64) {
        if (result.count >= CLIP_MAX_FACTORS - 1u) {
            result.invalid = 1u;
            return result;
        }
        result.values[result.count] = CLIP_SCALE_CHUNK;
        result.count += 1u;
        exponent += 64;
    }
    let last = ldexp(quotient.fract, exponent);
    if (result.count == 0u && abs(last - 1.0) <= CLIP_SCALE_EPSILON) { return result; }
    if ((bitcast<u32>(last) & 0x7f800000u) == 0x7f800000u || last <= 0.0 || last > 1.0) {
        result.invalid = 1u;
        return result;
    }
    result.values[result.count] = last;
    result.count += 1u;
    return result;
}
