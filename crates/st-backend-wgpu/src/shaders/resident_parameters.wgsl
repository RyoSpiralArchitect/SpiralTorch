SGD_CANDIDATE

struct Params {
    len: u32,
    groups_x: u32,
    count: u32,
    index: u32,
    rate: f32,
    padding0: u32,
    padding1: u32,
    padding2: u32,
};
@group(0) @binding(0) var<storage, read> old: array<u32>;
@group(0) @binding(1) var<storage, read> gradients: array<u32>;
@group(0) @binding(2) var<storage, read_write> candidates: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read> old_flags: array<u32>;
@group(0) @binding(5) var<storage, read> gradient_flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> output_flags: array<u32>;
@group(0) @binding(7) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> p: Params;

fn index(wid: vec3<u32>, lane: u32) -> u32 {
    return (wid.y * p.groups_x + wid.x) * 256u + lane;
}

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
        let candidate = sgd_candidate(bitcast<f32>(old[i]), bitcast<f32>(gradients[i]), p.rate, 4096u);
        candidates[i] = bitcast<u32>(candidate.value);
        if (candidate.flags != 0u) { atomicOr(&flags[p.index], candidate.flags); }
    }
}

@compute @workgroup_size(1)
fn decide() {
    var all = 0u;
    for (var i = 0u; i < p.count; i++) { all |= atomicLoad(&flags[i]); }
    atomicStore(&flags[p.count + 1u], all);
}

@compute @workgroup_size(256)
fn commit(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i == 0u) {
        var inherited = 0u;
        for (var j = 0u; j < arrayLength(&old_flags); j++) { inherited |= old_flags[j]; }
        output_flags[0] = inherited;
    }
    if (i < p.len) {
        if (atomicLoad(&flags[p.count + 1u]) == 0u && p.rate != 0.0) {
            output[i] = candidates[i];
        } else {
            // Copy integer bits so zero-rate/rejected updates preserve signed zero.
            output[i] = old[i];
        }
    }
}
