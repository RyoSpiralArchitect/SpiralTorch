// Private two-component significand with an extended binary exponent. This is
// not IEEE f64: it preserves cancellation and exponent range for LayerNorm,
// without requiring optional shader-f64 or flushing subnormal input bits.
struct Wide {
    hi: f32,
    lo: f32,
    exponent: i32,
    _pad: u32,
};

struct LayerNormRow {
    inverse_std: Wide,
    square_sum: Wide,
};

fn parts(value: f32) -> Wide {
    let bits = bitcast<u32>(value);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude == 0u) { return Wide(0.0, 0.0, 0, 0u); }
    let encoded = magnitude >> 23u;
    let fraction = magnitude & 0x7fffffu;
    let sign = select(1.0, -1.0, (bits >> 31u) != 0u);
    if (encoded == 0u) {
        let shift = 32u - countLeadingZeros(fraction);
        return Wide(sign * ldexp(f32(fraction), -i32(shift)), 0.0, i32(shift) - 149, 0u);
    }
    return Wide(sign * bitcast<f32>(0x3f000000u | fraction), 0.0, i32(encoded) - 126, 0u);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = rounded_add(a, b);
    let bv = rounded_add(s, -a);
    let error = rounded_add(rounded_add(a, -rounded_add(s, -bv)), rounded_add(b, -bv));
    return vec2<f32>(s, error);
}

fn wide_normalize(hi: f32, lo: f32, exponent: i32) -> Wide {
    let pair = two_sum(hi, lo);
    let leading = parts(pair.x);
    if (leading.hi == 0.0) { return leading; }
    let trailing = parts(pair.y);
    var low = 0.0;
    if (trailing.hi != 0.0) { low = align(trailing.hi, trailing.exponent - leading.exponent); }
    return Wide(leading.hi, low, exponent + leading.exponent, 0u);
}

fn align(value: f32, exponent: i32) -> f32 {
    if (exponent < -120) { return 0.0; }
    return ldexp(value, exponent);
}

fn wide_add(a: Wide, b: Wide) -> Wide {
    if (a.hi == 0.0) { return b; }
    if (b.hi == 0.0) { return a; }
    let exponent = max(a.exponent, b.exponent);
    let pa = vec2<f32>(align(a.hi, a.exponent - exponent), align(a.lo, a.exponent - exponent));
    let pb = vec2<f32>(align(b.hi, b.exponent - exponent), align(b.lo, b.exponent - exponent));
    let sum = two_sum(pa.x, pb.x);
    return wide_normalize(sum.x, rounded_add(sum.y, rounded_add(pa.y, pb.y)), exponent);
}

fn wide_neg(a: Wide) -> Wide { return Wide(-a.hi, -a.lo, a.exponent, 0u); }
fn wide_sub(a: Wide, b: Wide) -> Wide { return wide_add(a, wide_neg(b)); }

fn wide_mul(a: Wide, b: Wide) -> Wide {
    if (a.hi == 0.0 || b.hi == 0.0) { return parts(0.0); }
    // Exact 24 x 24 significand product in base 4096. Integer arithmetic also
    // prevents fast-math contraction from deleting a floating FMA residual.
    let ma = (bitcast<u32>(a.hi) & 0x7fffffu) | 0x800000u;
    let mb = (bitcast<u32>(b.hi) & 0x7fffffu) | 0x800000u;
    let low = (ma & 4095u) * (mb & 4095u);
    let middle = (ma >> 12u) * (mb & 4095u) + (ma & 4095u) * (mb >> 12u) + (low >> 12u);
    let high = (ma >> 12u) * (mb >> 12u) + (middle >> 12u);
    let tail = (low & 4095u) | ((middle & 4095u) << 12u);
    let sign = select(1.0, -1.0, (a.hi < 0.0) != (b.hi < 0.0));
    let h = sign * ldexp(f32(high), -24);
    let l = rounded_add(sign * ldexp(f32(tail), -48), rounded_add(a.hi * b.lo, a.lo * b.hi));
    return wide_normalize(h, l, a.exponent + b.exponent);
}

fn wide_div(a: Wide, b: Wide) -> Wide {
    if (a.hi == 0.0 || b.hi == 0.0) { return parts(0.0); }
    var quotient = parts(a.hi / b.hi);
    quotient.exponent += a.exponent - b.exponent;
    for (var refinement = 0u; refinement < 2u; refinement += 1u) {
        let remainder = wide_sub(a, wide_mul(quotient, b));
        var correction = parts(remainder.hi / b.hi);
        correction.exponent += remainder.exponent - b.exponent;
        var tail = parts(remainder.lo / b.hi);
        tail.exponent += remainder.exponent - b.exponent;
        quotient = wide_add(quotient, wide_add(correction, tail));
    }
    return quotient;
}

fn expansion_add(accumulator: vec4<f32>, value: f32) -> vec4<f32> {
    let a = two_sum(accumulator.x, value);
    let b = two_sum(accumulator.y, a.y);
    let c = two_sum(accumulator.z, b.y);
    return vec4<f32>(a.x, b.x, c.x, rounded_add(accumulator.w, c.y));
}

fn product_parts(a: f32, b: f32, exponent: i32, base_exp: i32) -> vec2<f32> {
    let product = wide_mul(parts(a), parts(b));
    return vec2<f32>(align(product.hi, product.exponent + exponent - base_exp),
                     align(product.lo, product.exponent + exponent - base_exp));
}

fn wide_difference_of_products(a: Wide, b: Wide, c: Wide, d: Wide) -> Wide {
    let ab = wide_mul(a, b);
    let cd = wide_mul(c, d);
    let difference = wide_sub(ab, cd);
    if (ab.hi == 0.0 || cd.hi == 0.0) { return difference; }
    let base_exp = max(a.exponent + b.exponent, c.exponent + d.exponent);
    // Two-component products retain about 48 significand bits. Recompute the
    // difference before rounding when cancellation would consume >20 bits.
    if (difference.hi != 0.0 && difference.exponent >= base_exp - 20) { return difference; }
    var expansion = vec4<f32>(0.0);
    let av = vec2<f32>(a.hi, a.lo);
    let bv = vec2<f32>(b.hi, b.lo);
    let cv = vec2<f32>(c.hi, c.lo);
    let dv = vec2<f32>(d.hi, d.lo);
    for (var i = 0u; i < 2u; i += 1u) {
        for (var j = 0u; j < 2u; j += 1u) {
            let p = product_parts(av[i], bv[j], a.exponent + b.exponent, base_exp);
            let q = product_parts(cv[i], dv[j], c.exponent + d.exponent, base_exp);
            expansion = expansion_add(expansion, p.x);
            expansion = expansion_add(expansion, -q.x);
            expansion = expansion_add(expansion, p.y);
            expansion = expansion_add(expansion, -q.y);
        }
    }
    var result = wide_add(wide_add(parts(expansion.x), parts(expansion.y)),
                          wide_add(parts(expansion.z), parts(expansion.w)));
    result.exponent += base_exp;
    return result;
}

fn wide_sqrt(a: Wide) -> Wide {
    if (a.hi == 0.0) { return a; }
    let odd = (a.exponent & 1) != 0;
    var root = parts(sqrt(a.hi * select(1.0, 2.0, odd)));
    root.exponent += a.exponent >> 1;
    var twice = root;
    twice.exponent += 1;
    return wide_add(root, wide_div(wide_sub(a, wide_mul(root, root)), twice));
}

fn wide_float(a: Wide) -> f32 {
    if (a.hi == 0.0) { return 0.0; }
    let bits = bitcast<u32>(a.hi);
    let sign = bits & 0x80000000u;
    let encoded = 126 + a.exponent;
    if (encoded >= 255) { return bitcast<f32>(sign | 0x7f800000u); }
    if (encoded > 0) { return bitcast<f32>(sign | (u32(encoded) << 23u) | (bits & 0x7fffffu)); }
    let distance = 1 - encoded;
    if (distance > 24) { return bitcast<f32>(sign); }
    let significand = (bits & 0x7fffffu) | 0x800000u;
    var rounded = significand >> u32(distance);
    let remainder = significand & ((1u << u32(distance)) - 1u);
    let half = 1u << u32(distance - 1);
    // The low component breaks exact halfway cases before the subnormal store.
    let directed_tail = a.lo * select(1.0, -1.0, a.hi < 0.0);
    if (remainder > half || (remainder == half && (directed_tail > 0.0 || (directed_tail == 0.0 && (rounded & 1u) != 0u)))) {
        rounded += 1u;
    }
    return bitcast<f32>(sign | rounded);
}
