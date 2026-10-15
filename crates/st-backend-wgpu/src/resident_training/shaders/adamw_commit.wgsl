@group(0) @binding(0) var<storage, read> old: array<u32>;
@group(0) @binding(1) var<storage, read> state: array<u32>;
@group(0) @binding(2) var<storage, read> candidates: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read_write> next_state: array<u32>;
@group(0) @binding(5) var<storage, read> old_flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> output_flags: array<u32>;
@group(0) @binding(7) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> p: Params;

@compute @workgroup_size(256)
fn commit(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    let accepted = atomicLoad(&flags[p.count + 1u]) == 0u && p.rate != 0.0;
    if (i == 0u) {
        var inherited = 0u;
        for (var j = 0u; j < arrayLength(&old_flags); j++) { inherited |= old_flags[j]; }
        output_flags[0] = inherited;
        next_state[2u*p.len] = select(state[2u*p.len], candidates[3u*p.len], accepted);
    }
    if (i < p.len) {
        output[i] = select(old[i], candidates[i], accepted);
        next_state[i] = select(state[i], candidates[p.len + i], accepted);
        next_state[p.len + i] = select(state[p.len + i], candidates[2u*p.len + i], accepted);
    }
}
