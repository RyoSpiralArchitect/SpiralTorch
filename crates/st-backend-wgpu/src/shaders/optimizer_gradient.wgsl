// Shared by clip-only SGD and fused EMA preparation. The Gain-role bit and
// explicit legacy policy are independent; only their conjunction normalizes.
fn effective(value: f32) -> f32 {
    check(value, 4096u);
    var scale = 1.0;
    if (p.gelu != 0u && step._pad0 != 0.0) { scale = 1.0 / f32(p.rows); }
    let result = value * scale;
    check(result, 4096u);
    return result;
}

// b holds the ordered normal-f32 scale factors; c holds the raw gradient.
fn clipped_gradient(i: u32) -> f32 {
    var gradient = effective(c[i]);
    let count = u32(b[0]);
    for (var j = 0u; j < count; j += 1u) { gradient *= b[1u + j]; }
    check(gradient, 4096u);
    return gradient;
}
