// Private three-component significand with an extended binary exponent. This is
// not IEEE f64: it preserves cancellation and exponent range for LayerNorm,
// without requiring optional shader-f64 or flushing subnormal input bits.
struct Wide {
    hi: f32,
    lo: f32,
    exponent: i32,
    tail: f32,
};

struct LayerNormRow {
    inverse_std: Wide,
    square_sum: Wide,
};

fn parts(value: f32) -> Wide {
    let bits = bitcast<u32>(value);
    let magnitude = bits & 0x7fffffffu;
    if (magnitude == 0u) { return Wide(0.0, 0.0, 0, 0.0); }
    let encoded = magnitude >> 23u;
    let fraction = magnitude & 0x7fffffu;
    let sign = select(1.0, -1.0, (bits >> 31u) != 0u);
    if (encoded == 0u) {
        let shift = 32u - countLeadingZeros(fraction);
        return Wide(sign * ldexp(f32(fraction), -i32(shift)), 0.0, i32(shift) - 149, 0.0);
    }
    return Wide(sign * bitcast<f32>(0x3f000000u | fraction), 0.0, i32(encoded) - 126, 0.0);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = rounded_add(a, b);
    let bv = rounded_add(s, -a);
    let error = rounded_add(rounded_add(a, -rounded_add(s, -bv)), rounded_add(b, -bv));
    return vec2<f32>(s, error);
}

fn wide_normalize(hi: f32, lo: f32, tail: f32, exponent: i32) -> Wide {
    let low = two_sum(lo, tail);
    let high = two_sum(hi, low.x);
    let rest = two_sum(high.y, low.y);
    let carry = two_sum(high.x, rest.x);
    let trailing = two_sum(carry.y, rest.y);
    let leading = parts(carry.x);
    if (leading.hi == 0.0) { return leading; }
    return Wide(leading.hi, align(trailing.x, -leading.exponent),
                exponent + leading.exponent, align(trailing.y, -leading.exponent));
}

fn align(value: f32, exponent: i32) -> f32 {
    if (exponent < -120) { return 0.0; }
    return ldexp(value, exponent);
}

fn wide_add(a: Wide, b: Wide) -> Wide {
    if (a.hi == 0.0) { return b; }
    if (b.hi == 0.0) { return a; }
    let exponent = max(a.exponent, b.exponent);
    let av = vec3<f32>(a.hi, a.lo, a.tail);
    let bv = vec3<f32>(b.hi, b.lo, b.tail);
    var expansion = vec4<f32>(0.0);
    for (var i = 0u; i < 3u; i += 1u) {
        expansion = expansion_add(expansion, align(av[i], a.exponent - exponent));
        expansion = expansion_add(expansion, align(bv[i], b.exponent - exponent));
    }
    return wide_from_expansion(expansion, exponent);
}

fn wide_neg(a: Wide) -> Wide { return Wide(-a.hi, -a.lo, a.exponent, -a.tail); }
fn wide_sub(a: Wide, b: Wide) -> Wide { return wide_add(a, wide_neg(b)); }

fn significand_product(a: f32, b: f32) -> vec2<f32> {
    // Exact 24 x 24 significand product in base 4096. Integer arithmetic also
    // prevents fast-math contraction from deleting a floating FMA residual.
    let ma = (bitcast<u32>(a) & 0x7fffffu) | 0x800000u;
    let mb = (bitcast<u32>(b) & 0x7fffffu) | 0x800000u;
    let low = (ma & 4095u) * (mb & 4095u);
    let middle = (ma >> 12u) * (mb & 4095u) + (ma & 4095u) * (mb >> 12u) + (low >> 12u);
    let high = (ma >> 12u) * (mb >> 12u) + (middle >> 12u);
    let tail = (low & 4095u) | ((middle & 4095u) << 12u);
    let sign = select(1.0, -1.0, (a < 0.0) != (b < 0.0));
    let h = sign * ldexp(f32(high), -24);
    return vec2<f32>(h, sign * ldexp(f32(tail), -48));
}

fn wide_mul(a: Wide, b: Wide) -> Wide {
    if (a.hi == 0.0 || b.hi == 0.0) { return parts(0.0); }
    var expansion = vec4<f32>(0.0);
    let av = vec3<f32>(a.hi, a.lo, a.tail);
    let bv = vec3<f32>(b.hi, b.lo, b.tail);
    for (var i = 0u; i < 3u; i += 1u) {
        for (var j = 0u; j < 3u; j += 1u) {
            let product = product_parts(av[i], bv[j], 0, 0);
            expansion = expansion_add(expansion, product.x);
            expansion = expansion_add(expansion, product.y);
        }
    }
    return wide_from_expansion(expansion, a.exponent + b.exponent);
}

fn wide_div(a: Wide, b: Wide) -> Wide {
    if (a.hi == 0.0 || b.hi == 0.0) { return parts(0.0); }
    var quotient = parts(a.hi / b.hi);
    quotient.exponent += a.exponent - b.exponent;
    for (var refinement = 0u; refinement < 2u; refinement += 1u) {
        let remainder = wide_sub(a, wide_mul(quotient, b));
        var correction = parts(remainder.hi / b.hi);
        correction.exponent += remainder.exponent - b.exponent;
        var low = parts(remainder.lo / b.hi);
        low.exponent += remainder.exponent - b.exponent;
        var tail = parts(remainder.tail / b.hi);
        tail.exponent += remainder.exponent - b.exponent;
        quotient = wide_add(quotient, wide_add(correction, wide_add(low, tail)));
    }
    return quotient;
}

fn expansion_add(accumulator: vec4<f32>, value: f32) -> vec4<f32> {
    let a = two_sum(accumulator.x, value);
    let b = two_sum(accumulator.y, a.y);
    let c = two_sum(accumulator.z, b.y);
    return vec4<f32>(a.x, b.x, c.x, rounded_add(accumulator.w, c.y));
}

fn wide_from_expansion(value: vec4<f32>, exponent: i32) -> Wide {
    let low = two_sum(value.z, value.w);
    let middle = two_sum(value.y, low.x);
    let high = two_sum(value.x, middle.x);
    let rest = two_sum(high.y, rounded_add(middle.y, low.y));
    return wide_normalize(high.x, rest.x, rest.y, exponent);
}

fn product_parts(a: f32, b: f32, exponent: i32, base_exp: i32) -> vec2<f32> {
    if (a == 0.0 || b == 0.0) { return vec2<f32>(0.0); }
    let pa = parts(a);
    let pb = parts(b);
    let product = significand_product(pa.hi, pb.hi);
    let shift = pa.exponent + pb.exponent + exponent - base_exp;
    return vec2<f32>(align(product.x, shift), align(product.y, shift));
}

fn wide_difference_of_products(a: Wide, b: Wide, c: Wide, d: Wide) -> Wide {
    let ab = wide_mul(a, b);
    let cd = wide_mul(c, d);
    let difference = wide_sub(ab, cd);
    if (ab.hi == 0.0 || cd.hi == 0.0) { return difference; }
    let base_exp = max(a.exponent + b.exponent, c.exponent + d.exponent);
    // Three-component products retain about 72 significand bits. Recompute the
    // difference before rounding when cancellation would consume >44 bits.
    if (difference.hi != 0.0 && difference.exponent >= base_exp - 44) { return difference; }
    var expansion = vec4<f32>(0.0);
    let av = vec3<f32>(a.hi, a.lo, a.tail);
    let bv = vec3<f32>(b.hi, b.lo, b.tail);
    let cv = vec3<f32>(c.hi, c.lo, c.tail);
    let dv = vec3<f32>(d.hi, d.lo, d.tail);
    for (var i = 0u; i < 3u; i += 1u) {
        for (var j = 0u; j < 3u; j += 1u) {
            let p = product_parts(av[i], bv[j], a.exponent + b.exponent, base_exp);
            let q = product_parts(cv[i], dv[j], c.exponent + d.exponent, base_exp);
            expansion = expansion_add(expansion, p.x);
            expansion = expansion_add(expansion, -q.x);
            expansion = expansion_add(expansion, p.y);
            expansion = expansion_add(expansion, -q.y);
        }
    }
    return wide_from_expansion(expansion, base_exp);
}

fn wide_sqrt(a: Wide) -> Wide {
    if (a.hi == 0.0) { return a; }
    let odd = (a.exponent & 1) != 0;
    var root = parts(sqrt(a.hi * select(1.0, 2.0, odd)));
    root.exponent += a.exponent >> 1;
    for (var refinement = 0u; refinement < 2u; refinement += 1u) {
        var twice = root;
        twice.exponent += 1;
        root = wide_add(root, wide_div(wide_sub(a, wide_mul(root, root)), twice));
    }
    return root;
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
    let directed_tail = select(a.tail, a.lo, a.lo != 0.0) * select(1.0, -1.0, a.hi < 0.0);
    if (remainder > half || (remainder == half && (directed_tail > 0.0 || (directed_tail == 0.0 && (rounded & 1u) != 0u)))) {
        rounded += 1u;
    }
    return bitcast<f32>(sign | rounded);
}
