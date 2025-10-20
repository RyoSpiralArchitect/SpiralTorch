struct FlipParams {
    height: u32,
    width: u32,
    channels: u32,
    apply: u32,
};

@group(0) @binding(0)
var<storage, read> input_image: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_image: array<f32>;

@group(0) @binding(2)
var<uniform> params: FlipParams;

fn index(channel: u32, y: u32, x: u32, width: u32, height: u32) -> u32 {
    return channel * width * height + y * width + x;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.z >= params.channels || gid.y >= params.height || gid.x >= params.width {
        return;
    }
    let c = gid.z;
    let y = gid.y;
    let src_x = select(gid.x, params.width - 1u - gid.x, params.apply == 1u);
    let value = input_image[index(c, y, src_x, params.width, params.height)];
    let dst_idx = index(c, y, gid.x, params.width, params.height);
    output_image[dst_idx] = value;
}
