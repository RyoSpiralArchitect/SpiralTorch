// Saturated derivative shared with checked host/tensor semantics.
fn gelu_prime(x: f32) -> f32 {
    if (abs(x) >= 10.0) { return select(0.0, 1.0, x > 0.0); }
    let square = x*x;
    let inner = 0.7978846 * (x + 0.044715*x*square);
    let t = tanh(clamp(inner, -10.0, 10.0));
    return 0.5*(1.0+t) + 0.5*x*(1.0-t*t)*0.7978846*(1.0+3.0*0.044715*square);
}
