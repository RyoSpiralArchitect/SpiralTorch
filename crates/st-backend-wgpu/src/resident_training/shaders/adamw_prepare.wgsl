@group(0) @binding(0) var<storage, read> old: array<u32>;
@group(0) @binding(1) var<storage, read> gradients: array<u32>;
@group(0) @binding(2) var<storage, read> state: array<u32>;
@group(0) @binding(3) var<storage, read_write> candidates: array<u32>;
@group(0) @binding(4) var<storage, read> old_flags: array<u32>;
@group(0) @binding(5) var<storage, read> gradient_flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p: Params;

@compute @workgroup_size(256)
fn prepare(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i == 0u) {
        var inherited = 0u;
        for (var j = 0u; j < arrayLength(&old_flags); j++) { inherited |= old_flags[j]; }
        for (var j = 0u; j < arrayLength(&gradient_flags); j++) { inherited |= gradient_flags[j]; }
        if (inherited != 0u) { atomicOr(&flags[p.index], INVALID_TENSOR_FLAG); }
    }
    if (i < p.len) {
        let c = adamw_candidate(bitcast<f32>(old[i]), bitcast<f32>(gradients[i]),
            bitcast<f32>(state[i]), bitcast<f32>(state[p.len + i]), state[2u*p.len],
            p.rate, p.decay, p.beta1, p.beta2, p.epsilon);
        candidates[i] = bitcast<u32>(c.value);
        candidates[p.len + i] = bitcast<u32>(c.first);
        candidates[2u*p.len + i] = bitcast<u32>(c.second);
        if (i == 0u) { candidates[3u*p.len] = c.steps; }
        if (c.flags != 0u) { atomicOr(&flags[p.index], c.flags); }
    }
}
@compute @workgroup_size(1)
fn decide() {
    var all = 0u;
    for (var i = 0u; i < p.count; i++) { all |= atomicLoad(&flags[i]); }
    atomicStore(&flags[p.count + 1u], all);
}
