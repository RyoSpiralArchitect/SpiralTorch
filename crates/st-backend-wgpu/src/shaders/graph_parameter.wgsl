// Graph-owned parameter candidate and effective optimizer gradient.
// p.gelu selects a legacy row-average parameter; step._pad0 enables the policy.
@compute @workgroup_size(256)
fn prepare_parameter(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i < p.len) {
        var scale = 1.0;
        if (p.gelu != 0u && step._pad0 != 0.0) { scale = 1.0 / f32(p.rows); }
        let gradient = c[i] * scale;
        let candidate = sgd_candidate(a[i], gradient, step.rate, 4096u);
        check(c[i], 4096u);
        if (candidate.flags != 0u) { atomicOr(&validation[p.stage], candidate.flags); }
        out[i] = candidate.value;
        aux[i] = gradient;
    }
}
