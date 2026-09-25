struct BatchFlipParams {
    height: u32,
    width: u32,
    channels_per_image: u32,
    batch_size: u32,
};

@group(0) @binding(0)
var<storage, read> input_image: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_image: array<f32>;

@group(0) @binding(2)
var<uniform> params: BatchFlipParams;

@group(0) @binding(3)
var<storage, read> flip_mask: array<u32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total_channels = params.batch_size * params.channels_per_image;
    if gid.z >= total_channels || gid.y >= params.height || gid.x >= params.width {
        return;
    }
    let image = gid.z / params.channels_per_image;
    let src_x = select(gid.x, params.width - 1u - gid.x, flip_mask[image] != 0u);
    let plane_offset = gid.z * params.height * params.width + gid.y * params.width;
    output_image[plane_offset + gid.x] = input_image[plane_offset + src_x];
}
