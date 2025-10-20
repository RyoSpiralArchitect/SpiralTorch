struct CropParams {
    src_height: u32,
    src_width: u32,
    dst_height: u32,
    dst_width: u32,
    top: u32,
    left: u32,
    channels: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<storage, read> input_image: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_image: array<f32>;

@group(0) @binding(2)
var<uniform> params: CropParams;

fn index(channel: u32, y: u32, x: u32, width: u32, height: u32) -> u32 {
    return channel * width * height + y * width + x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.z >= params.channels || gid.y >= params.dst_height || gid.x >= params.dst_width {
        return;
    }
    let src_y = gid.y + params.top;
    let src_x = gid.x + params.left;
    let c = gid.z;
    let src_idx = index(c, src_y, src_x, params.src_width, params.src_height);
    let dst_idx = index(c, gid.y, gid.x, params.dst_width, params.dst_height);
    output_image[dst_idx] = input_image[src_idx];
}
