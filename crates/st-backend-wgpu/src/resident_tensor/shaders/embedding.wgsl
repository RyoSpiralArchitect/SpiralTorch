struct Params {
    cols: u32, tokens: u32, rows: u32, len: u32,
    groups_x: u32, groups: u32, input_offset: u32, flag_words: u32,
};
@group(0) @binding(0) var<storage, read> input: array<f32>;
// IDs, then V+1 CSR offsets, then token positions in stable input order.
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> input_flags: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_flags: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

ROUNDED_ADD

fn inherit(index: u32) {
    if (index != 0u) { return; }
    var failed = 0u;
    for (var flag = 0u; flag < params.flag_words; flag++) { failed |= input_flags[flag]; }
    if (failed != 0u) { atomicOr(&output_flags[0], INVALID_TENSOR_FLAG); }
}

@compute @workgroup_size(256)
fn gather(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group = wid.y * params.groups_x + wid.x;
    if (group >= params.groups) { return; }
    let i = group * 256u + lane;
    inherit(i);
    if (i >= params.len) { return; }
    let row = indices[i / params.cols];
    output[i] = input[params.input_offset + row * params.cols + i % params.cols];
}

@compute @workgroup_size(256)
fn pullback(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group = wid.y * params.groups_x + wid.x;
    if (group >= params.groups) { return; }
    let i = group * 256u + lane;
    inherit(i);
    if (i >= params.len) { return; }
    let row = i / params.cols;
    let col = i % params.cols;
    var bits = 0u;
    for (var slot = indices[params.tokens + row]; slot < indices[params.tokens + row + 1u]; slot++) {
        let token = indices[params.tokens + params.rows + 1u + slot];
        bits = rounded_add_bits(bits, bitcast<u32>(input[params.input_offset + token * params.cols + col]));
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            atomicOr(&output_flags[0], INVALID_TENSOR_FLAG);
            bits = 0u;
        }
    }
    output[i] = bitcast<f32>(bits);
}
