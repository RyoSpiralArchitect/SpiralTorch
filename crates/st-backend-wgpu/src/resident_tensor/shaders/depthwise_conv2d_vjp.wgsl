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
    target_len: u32,
    groups_x: u32,
    batch: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
};

@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> weight_values: array<f32>;
@group(0) @binding(2) var<storage, read> upstream_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> gradient_values: array<f32>;
@group(0) @binding(4) var<storage, read> input_flags: array<u32>;
@group(0) @binding(5) var<storage, read> weight_flags: array<u32>;
@group(0) @binding(6) var<storage, read> upstream_flags: array<u32>;
@group(0) @binding(7) var<storage, read_write> gradient_flags: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: Params;

fn index_for(group: vec3<u32>, lane: u32) -> u32 {
    return (group.y * params.groups_x + group.x) * 64u + lane;
}

fn inherit_guard(index: u32) {
    if (index == 0u) {
        atomicOr(&gradient_flags[0], input_flags[0] | weight_flags[0] | upstream_flags[0]);
    }
}

fn write_gradient(index: u32, value: f32) {
    gradient_values[index] = value;
    if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&gradient_flags[0], 0x80000000u);
    }
}

@compute @workgroup_size(64)
fn input_vjp(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = index_for(group, lane);
    inherit_guard(index);
    if (index >= params.target_len) { return; }

    let input_spatial = params.input_h * params.input_w;
    let output_spatial = params.output_h * params.output_w;
    let batch = index / (params.channels * input_spatial);
    let channel = (index / input_spatial) % params.channels;
    let input_y = (index % input_spatial) / params.input_w;
    let input_x = index % params.input_w;
    var sum = 0.0;
    for (var ky = 0u; ky < params.kernel_h; ky = ky + 1u) {
        let y_numerator = i32(input_y) + params.pad_h - i32(ky * params.dilation_h);
        if (y_numerator >= 0 && u32(y_numerator) % params.stride_h == 0u) {
            let output_y = u32(y_numerator) / params.stride_h;
            if (output_y < params.output_h) {
                for (var kx = 0u; kx < params.kernel_w; kx = kx + 1u) {
                    let x_numerator = i32(input_x) + params.pad_w - i32(kx * params.dilation_w);
                    if (x_numerator >= 0 && u32(x_numerator) % params.stride_w == 0u) {
                        let output_x = u32(x_numerator) / params.stride_w;
                        if (output_x < params.output_w) {
                            let upstream_index = (batch * params.channels + channel) * output_spatial
                                + output_y * params.output_w + output_x;
                            let weight_index = channel * params.span + ky * params.kernel_w + kx;
                            sum = sum + upstream_values[upstream_index] * weight_values[weight_index];
                        }
                    }
                }
            }
        }
    }
    write_gradient(index, sum);
}

@compute @workgroup_size(64)
fn weight_vjp(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = index_for(group, lane);
    inherit_guard(index);
    if (index >= params.target_len) { return; }

    let channel = index / params.span;
    let kernel_y = (index % params.span) / params.kernel_w;
    let kernel_x = index % params.kernel_w;
    let output_spatial = params.output_h * params.output_w;
    var sum = 0.0;
    for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
        for (var output_y = 0u; output_y < params.output_h; output_y = output_y + 1u) {
            let input_y = i32(output_y * params.stride_h + kernel_y * params.dilation_h) - params.pad_h;
            if (input_y >= 0 && input_y < i32(params.input_h)) {
                for (var output_x = 0u; output_x < params.output_w; output_x = output_x + 1u) {
                    let input_x = i32(output_x * params.stride_w + kernel_x * params.dilation_w) - params.pad_w;
                    if (input_x >= 0 && input_x < i32(params.input_w)) {
                        let input_index = ((batch * params.channels + channel) * params.input_h
                            + u32(input_y)) * params.input_w + u32(input_x);
                        let upstream_index = (batch * params.channels + channel) * output_spatial
                            + output_y * params.output_w + output_x;
                        sum = sum + input_values[input_index] * upstream_values[upstream_index];
                    }
                }
            }
        }
    }
    write_gradient(index, sum);
}

@compute @workgroup_size(64)
fn bias_vjp(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = index_for(group, lane);
    inherit_guard(index);
    if (index >= params.target_len) { return; }

    let output_spatial = params.output_h * params.output_w;
    var sum = 0.0;
    for (var batch = 0u; batch < params.batch; batch = batch + 1u) {
        for (var position = 0u; position < output_spatial; position = position + 1u) {
            let upstream_index = (batch * params.channels + index) * output_spatial + position;
            sum = sum + upstream_values[upstream_index];
        }
    }
    write_gradient(index, sum);
}
