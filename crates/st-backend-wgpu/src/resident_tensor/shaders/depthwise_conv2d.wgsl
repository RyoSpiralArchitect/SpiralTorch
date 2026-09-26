struct Params {
    channels: u32,
    input_h: u32,
    input_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: i32,
    pad_w: i32,
    dilation_h: u32,
    dilation_w: u32,
    output_h: u32,
    output_w: u32,
    span: u32,
    output_len: u32,
    groups_x: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> weight_values: array<f32>;
@group(0) @binding(2) var<storage, read> bias_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(4) var<storage, read> input_flags: array<u32>;
@group(0) @binding(5) var<storage, read> weight_flags: array<u32>;
@group(0) @binding(6) var<storage, read> bias_flags: array<u32>;
@group(0) @binding(7) var<storage, read_write> output_flags: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = (group.y * params.groups_x + group.x) * 64u + lane;
    if (index == 0u) {
        let inherited = input_flags[0] | weight_flags[0] | bias_flags[0];
        atomicOr(&output_flags[0], inherited);
    }
    if (index >= params.output_len) { return; }

    let output_spatial = params.output_h * params.output_w;
    let output_cols = params.channels * output_spatial;
    let batch = index / output_cols;
    let channel = (index / output_spatial) % params.channels;
    let position = index % output_spatial;
    let output_y = position / params.output_w;
    let output_x = position % params.output_w;
    var value = bias_values[channel];

    for (var ky = 0u; ky < params.kernel_h; ky = ky + 1u) {
        for (var kx = 0u; kx < params.kernel_w; kx = kx + 1u) {
            let input_y = i32(output_y * params.stride_h + ky * params.dilation_h) - params.pad_h;
            let input_x = i32(output_x * params.stride_w + kx * params.dilation_w) - params.pad_w;
            if (input_y >= 0 && input_y < i32(params.input_h)
                && input_x >= 0 && input_x < i32(params.input_w)) {
                let input_index = ((batch * params.channels + channel) * params.input_h
                    + u32(input_y)) * params.input_w + u32(input_x);
                let weight_index = channel * params.span + ky * params.kernel_w + kx;
                value = value + input_values[input_index] * weight_values[weight_index];
            }
        }
    }
    output_values[index] = value;
    if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&output_flags[0], 0x80000000u);
    }
}
