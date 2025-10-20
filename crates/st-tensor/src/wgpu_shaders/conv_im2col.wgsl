struct Params {
    dims0: vec4<u32>;
    dims1: vec4<u32>;
    dims2: vec4<i32>;
    dims3: vec4<u32>;
    dims4: vec4<u32>;
};

@group(0) @binding(0)
var<storage, read> input_tensor: array<f32>;

@group(0) @binding(1)
var<storage, read_write> patch_tensor: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn row_offset(batch: u32, spatial: u32, row: u32, kernel_elems: u32) -> u32 {
    return (batch * spatial + row) * kernel_elems;
}

fn input_index(batch: u32, channel: u32, ih: u32, iw: u32, row_stride: u32, channel_stride: u32, width: u32) -> u32 {
    return batch * row_stride + channel * channel_stride + ih * width + iw;
}

@compute @workgroup_size(16, 4, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let kernel_elems = params.dims3.z;
    let spatial = params.dims3.w;
    let batch = params.dims0.x;

    if (global_id.x >= kernel_elems || global_id.y >= spatial || global_id.z >= batch) {
        return;
    }

    let input_h = params.dims0.z;
    let input_w = params.dims0.w;

    let kernel_h = params.dims1.x;
    let kernel_w = params.dims1.y;
    let stride_h = params.dims1.z;
    let stride_w = params.dims1.w;

    let pad_h = params.dims2.x;
    let pad_w = params.dims2.y;
    let dilation_h = params.dims2.z;
    let dilation_w = params.dims2.w;

    let out_w = params.dims3.y;

    let row_stride = params.dims4.x;
    let channel_stride = params.dims4.y;

    let row = global_id.y;
    let batch_index = global_id.z;
    let col = global_id.x;

    let oh = row / out_w;
    let ow = row % out_w;

    let kernel_span = kernel_h * kernel_w;
    let channel = col / kernel_span;
    let kernel_idx = col % kernel_span;
    let kh = kernel_idx / kernel_w;
    let kw = kernel_idx % kernel_w;

    let in_h = i32(oh * stride_h + kh * dilation_h) - pad_h;
    let in_w = i32(ow * stride_w + kw * dilation_w) - pad_w;

    var value: f32 = 0.0;
    if (in_h >= 0 && in_w >= 0 && in_h < i32(input_h) && in_w < i32(input_w)) {
        let ih = u32(in_h);
        let iw = u32(in_w);
        let input_idx = input_index(batch_index, channel, ih, iw, row_stride, channel_stride, input_w);
        value = input_tensor[input_idx];
    }

    let patch_idx = row_offset(batch_index, spatial, row, kernel_elems) + col;
    patch_tensor[patch_idx] = value;
}
