struct AdamWCandidate {
    value: f32,
    first: f32,
    second: f32,
    steps: u32,
    flags: u32,
};
fn adamw_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}
// Mirror Rust pre-operation bounds: WGSL overflow can return a finite value.
fn adamw_mul(a: f32, b: f32, valid: ptr<function, bool>) -> f32 {
    if (!*valid) { return 0.0; }
    let smaller = min(abs(a), abs(b));
    if (smaller > 1.0) {
        // The guard's own divisor must stay within WGSL's accuracy domain.
        if (smaller > bitcast<f32>(0x7e800000u)) {
            *valid = false; return 0.0;
        }
        if (max(abs(a), abs(b)) >= bitcast<f32>(0x7f7ffff0u) / smaller) {
            *valid = false; return 0.0;
        }
    }
    return a*b;
}
fn adamw_add(a: f32, b: f32, valid: ptr<function, bool>) -> f32 {
    if (!*valid) { return 0.0; }
    if (a != 0.0 && b != 0.0 && (a > 0.0) == (b > 0.0)
        && abs(a) >= bitcast<f32>(0x7f7ffff0u) - abs(b)) {
        *valid = false; return 0.0;
    }
    return a+b;
}
fn adamw_div(a: f32, b: f32, valid: ptr<function, bool>) -> f32 {
    if (!*valid) { return 0.0; }
    if ((bitcast<u32>(b) & 0x7f800000u) == 0u || b <= 0.0
        || b > bitcast<f32>(0x7e800000u)
        || abs(a) >= bitcast<f32>(0x7f7ffff0u)
        || (b < 1.0 && abs(a) >= bitcast<f32>(0x7f7ffff0u)*b)) {
        *valid = false; return 0.0;
    }
    return a/b;
}
fn adamw_unit_powi(base: f32, exponent: u32) -> f32 {
    var b = base;
    var n = exponent;
    var value = 1.0;
    loop {
        if (n == 0u) { break; }
        if ((n & 1u) != 0u) { value *= b; }
        n >>= 1u;
        if (n != 0u) { b *= b; }
    }
    return value;
}
fn adamw_candidate(value: f32, gradient: f32, first: f32, second: f32, steps: u32,
                   rate: f32, decay: f32, beta1: f32, beta2: f32, epsilon: f32) -> AdamWCandidate {
    var out = AdamWCandidate(value, first, second, steps, 0u);
    if (!adamw_finite(value) || !adamw_finite(gradient) || !adamw_finite(first)
        || !adamw_finite(second) || second < 0.0) {
        out.flags = 32768u;
        return out;
    }
    if (rate == 0.0) { return out; }
    if (steps == 0xffffffffu) { out.flags = 131072u; return out; }
    out.steps = steps + 1u;
    var valid = true;
    let square = adamw_mul(gradient, gradient, &valid);
    let first_old = adamw_mul(beta1, first, &valid);
    let first_new = adamw_mul(1.0-beta1, gradient, &valid);
    out.first = adamw_add(first_old, first_new, &valid);
    let second_old = adamw_mul(beta2, second, &valid);
    let second_new = adamw_mul(1.0-beta2, square, &valid);
    out.second = adamw_add(second_old, second_new, &valid);
    let m_hat = adamw_div(out.first, 1.0-adamw_unit_powi(beta1, out.steps), &valid);
    let v_hat = adamw_div(out.second, 1.0-adamw_unit_powi(beta2, out.steps), &valid);
    let denominator = adamw_add(sqrt(v_hat), epsilon, &valid);
    let decay_amount = adamw_mul(rate, decay, &valid);
    let factor = adamw_add(1.0, -decay_amount, &valid);
    let decayed = adamw_mul(value, factor, &valid);
    let direction = adamw_div(m_hat, denominator, &valid);
    let change = adamw_mul(rate, direction, &valid);
    out.value = adamw_add(decayed, -change, &valid);
    if (!valid || !adamw_finite(square) || !adamw_finite(out.first) || !adamw_finite(out.second)
        || !adamw_finite(m_hat) || !adamw_finite(v_hat) || !adamw_finite(denominator)
        || denominator <= 0.0 || !adamw_finite(decay_amount) || !adamw_finite(decayed)
        || !adamw_finite(change) || !adamw_finite(out.value)) {
        out.flags = 65536u;
    }
    return out;
}
