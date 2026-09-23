// Saturated derivative shared with checked host/tensor semantics.
fn gelu_prime(x: f32) -> f32 {
    if (abs(x) >= 10.0) { return select(0.0, 1.0, x > 0.0); }
    let square = x*x;
    let inner = 0.7978846 * (x + 0.044715*x*square);
    // Keep the established central-domain evaluation order used in training.
    // At |x| <= 3, tanh(inner) is still separated from its saturated limit.
    if (abs(x) <= 3.0) {
        let t = tanh(inner);
        return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t*t) * 0.7978846 * (1.0 + 3.0 * 0.044715 * square);
    }
    // Compute (1+tanh(inner))/2 and sech(inner)^2 without subtracting
    // nearly equal values or amplifying a rounded tanh tail near +/-1.
    let q = exp(-2.0 * abs(inner));
    let inverse = 1.0 / (1.0 + q);
    let cdf = select(q * inverse, inverse, inner >= 0.0);
    let slope = 0.7978846 * (1.0 + 3.0 * 0.044715 * square);
    return cdf + 2.0 * x * slope * q * inverse * inverse;
}
