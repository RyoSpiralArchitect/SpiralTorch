struct ResizeParams {
    src_height: u32,
    src_width: u32,
    dst_height: u32,
    dst_width: u32,
    channels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var<storage, read> input_image: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_image: array<f32>;

@group(0) @binding(2)
var<uniform> params: ResizeParams;

fn index(channel: u32, y: u32, x: u32, width: u32, height: u32) -> u32 {
    return channel * width * height + y * width + x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.z >= params.channels || gid.y >= params.dst_height || gid.x >= params.dst_width {
        return;
    }
    let scale_y = f32(params.src_height) / f32(params.dst_height);
    let scale_x = f32(params.src_width) / f32(params.dst_width);
    let src_y = (f32(gid.y) + 0.5) * scale_y - 0.5;
    let src_x = (f32(gid.x) + 0.5) * scale_x - 0.5;
    let y0 = clamp(floor(src_y), 0.0, f32(params.src_height - 1));
    let x0 = clamp(floor(src_x), 0.0, f32(params.src_width - 1));
    let y1 = min(u32(y0) + 1u, params.src_height - 1u);
    let x1 = min(u32(x0) + 1u, params.src_width - 1u);
    let ly = src_y - f32(u32(y0));
    let lx = src_x - f32(u32(x0));

    let c = gid.z;
    let top_left = input_image[index(c, u32(y0), u32(x0), params.src_width, params.src_height)];
    let top_right = input_image[index(c, u32(y0), x1, params.src_width, params.src_height)];
    let bottom_left = input_image[index(c, y1, u32(x0), params.src_width, params.src_height)];
    let bottom_right = input_image[index(c, y1, x1, params.src_width, params.src_height)];

    let top = top_left * (1.0 - lx) + top_right * lx;
    let bottom = bottom_left * (1.0 - lx) + bottom_right * lx;
    let value = top * (1.0 - ly) + bottom * ly;

    let out_idx = index(c, gid.y, gid.x, params.dst_width, params.dst_height);
    output_image[out_idx] = value;
}
