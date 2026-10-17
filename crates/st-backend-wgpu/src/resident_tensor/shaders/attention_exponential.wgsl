// Shared by forward and VJP. Keep tiny weights in extended exponent form until
// after multiplication by V/upstream, rather than flushing an f32 probability.
fn score_weight(score: f32, maximum: f32) -> Wide {
    let delta = wide_sub(parts(score), parts(maximum));
    if (delta.hi == 0.0) { return parts(1.0); }
    if (delta.exponent >= 11) { return parts(0.0); }
    let value = wide_float(delta);
    if (value < -745.0) { return parts(0.0); }
    let exponent = i32(floor(value * 1.4426950408889634));
    let ln_two = wide_add(parts(0.6931471824645996), parts(-1.904654323148236e-9));
    let remainder = wide_sub(delta, wide_mul(parts(f32(exponent)), ln_two));
    var weight = parts(exp(wide_float(remainder)));
    weight.exponent += exponent;
    return weight;
}
