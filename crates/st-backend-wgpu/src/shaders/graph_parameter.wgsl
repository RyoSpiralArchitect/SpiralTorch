// Graph-owned parameter candidate and effective optimizer gradient.
// p.gelu is the Gain-role bit here; step._pad0 is the explicit legacy policy.
@compute @workgroup_size(256)
fn prepare_parameter(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i < p.len) {
        var scale = 1.0;
        if (p.gelu != 0u && step._pad0 != 0.0) { scale = 1.0 / f32(p.rows); }
        let gradient = c[i] * scale;
        let change = step.rate * gradient;
        let candidate = a[i] - change;
        check(c[i], 4096u); check(gradient, 4096u);
        check(change, 8192u); check(candidate, 16384u);
        out[i] = candidate;
        aux[i] = gradient;
    }
}
