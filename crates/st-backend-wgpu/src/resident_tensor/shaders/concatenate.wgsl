@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> params: array<u32>;
@group(0) @binding(3) var<storage, read> input_flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_flags: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group = wid.y * params[2] + wid.x;
    if (group >= params[3]) { return; }
    let i = group * 256u + lane;
    if (i == 0u) {
        var inherited = 0u;
        for (var flag = 0u; flag < params[6]; flag++) { inherited |= input_flags[flag]; }
        if (inherited != 0u) { atomicOr(&output_flags[0], INVALID_TENSOR_FLAG); }
    }
    if (i >= params[0]) { return; }
    let rank = params[1];
    var logical = i;
    var source = params[4];
    var destination = params[5];
    for (var axis = rank; axis > 0u; axis--) {
        let d = axis - 1u;
        let coordinate = logical % params[7u + d];
        logical /= params[7u + d];
        source += coordinate * params[7u + rank + d];
        destination += coordinate * params[7u + 2u * rank + d];
    }
    output[destination] = input[source];
}
