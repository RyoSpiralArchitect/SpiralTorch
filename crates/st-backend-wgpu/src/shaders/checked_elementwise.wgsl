fn check(x: f32) {
    if ((bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&flags[CHECKED_FLAG_INDEX], INVALID_TENSOR_FLAG);
    }
}

fn checked_apply(op: u32, x: f32, y: f32) -> f32 {
    check(x);
    var value = x;
    switch op {
        case OP_ADD: { check(y); value = x + y; }
        case OP_MULTIPLY: { check(y); value = x * y; }
        case OP_RELU: { value = max(x, 0.0); }
        case OP_GELU: {
            let square = x * x;
            let cubic = square * x;
            let inner = 0.7978846 * (x + 0.044715 * cubic);
            check(square); check(cubic); check(inner);
            value = 0.5 * x * (1.0 + tanh(clamp(inner, -10.0, 10.0)));
        }
        default: {}
    }
    check(value);
    return value;
}
