// Topos EMA history and parameters share one global candidate/commit decision.
// This path has a private Step uniform: _pad0 carries the EMA damping.
@compute @workgroup_size(256)
fn momentum_gradient(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i < p.len) {
        var scale = 1.0;
        if (p.gelu != 0u) { scale = 1.0 / f32(p.rows); }
        check(a[i], 4096u);
        let g = a[i] * scale;
        check(g, 4096u);
        out[i] = g;
    }
}

@compute @workgroup_size(256)
fn prepare_momentum(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i < p.len) {
        check(c[i], 4096u); check(b[i], 32768u);
        let next = ema_momentum(c[i], b[i], step._pad0);
        let change = step.rate * next;
        let candidate = a[i] - change;
        check(next, 32768u); check(change, 8192u); check(candidate, 16384u);
        out[i] = candidate;
        aux[i] = next;
    }
}

@compute @workgroup_size(256)
fn commit_momentum(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if (atomicLoad(&validation[p.stages + 1u]) != 0u || step.rate == 0.0) { return; }
    let i = index(wid, lane);
    if (i < p.len) {
        out[i] = a[i];
        aux[i] = b[i];
    }
}
