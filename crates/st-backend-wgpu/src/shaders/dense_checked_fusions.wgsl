// High-level inference preserves finite-value guards without host round trips.
@group(0) @binding(7) var<storage, read_write> validation: array<atomic<u32>>;

fn record_nonfinite(value: f32, flag: u32) {
    if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&validation[params.validation_index], flag | params.validation_mask);
    }
}

fn apply_fusions(acc: f32, index: u32, col: u32) -> f32 {
    var value = acc;
    if ((params.flags & FLAG_USE_BIAS) != 0u) { value = value + bias[col]; }
    if ((params.flags & FLAG_FUSED_RESIDUAL) != 0u) { value = value + residual[index]; }
    record_nonfinite(value, 1u);
    if ((params.flags & FLAG_FUSED_GELU) != 0u) {
        let x = value;
        let square = x * x;
        record_nonfinite(square, 2u);
        let cubic = square * x;
        record_nonfinite(cubic, 4u);
        let inner_arg = x + 0.044715 * cubic;
        record_nonfinite(inner_arg, 8u);
        let inner = 0.7978845834732056 * inner_arg;
        record_nonfinite(inner, 16u);
        // tanh is already +/-1 in f32 here. Some GPU implementations overflow
        // internally for large finite arguments; preserve the guards above.
        let t = tanh(clamp(inner, -10.0, 10.0));
        record_nonfinite(t, 32u);
        value = 0.5 * x * (1.0 + t);
    } else if ((params.flags & FLAG_FUSED_RELU) != 0u) {
        value = max(value, 0.0);
    }
    value = value * params.output_scale;
    record_nonfinite(value, 64u);
    return value;
}
