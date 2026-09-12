// Topos EMA history and parameters share one global candidate/commit decision.
// Private Step: _pad0 is the policy, _pad1 is damping, _pad2 enables clipping.

@compute @workgroup_size(256)
fn prepare_momentum(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i < p.len) {
        var gradient: f32;
        if (step._pad2 != 0.0) { gradient = clipped_gradient(i); }
        else { gradient = effective(c[i]); }
        check(d[i], 32768u);
        let next = ema_momentum(gradient, d[i], step._pad1);
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
