struct ColorJitterParams {
    dims: vec4<u32>,
    factors: vec4<f32>,
    means: vec4<f32>,
};

@group(0) @binding(0)
var<storage, read> input_image: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_image: array<f32>;

@group(0) @binding(2)
var<uniform> params: ColorJitterParams;

fn index(channel: u32, y: u32, x: u32, width: u32, height: u32) -> u32 {
    return channel * width * height + y * width + x;
}

fn apply_saturation(rgb: vec3<f32>, factor: f32) -> vec3<f32> {
    if factor == 1.0 {
        return rgb;
    }
    let gray = dot(rgb, vec3<f32>(0.29899597, 0.587096, 0.11390703));
    let base = vec3<f32>(gray, gray, gray);
    return (rgb - base) * factor + base;
}

fn apply_hue(rgb: vec3<f32>, radians: f32) -> vec3<f32> {
    if radians == 0.0 {
        return rgb;
    }
    let cos_h = cos(radians);
    let sin_h = sin(radians);
    let y = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    let u = dot(rgb, vec3<f32>(-0.14713, -0.28886, 0.436));
    let v = dot(rgb, vec3<f32>(0.615, -0.51499, -0.10001));
    let u_prime = u * cos_h - v * sin_h;
    let v_prime = u * sin_h + v * cos_h;
    return vec3<f32>(
        y + 1.13983 * v_prime,
        y - 0.39465 * u_prime - 0.58060 * v_prime,
        y + 2.03211 * u_prime,
    );
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let height = params.dims.x;
    let width = params.dims.y;
    let channels = params.dims.z;
    if gid.z >= channels || gid.y >= height || gid.x >= width {
        return;
    }

    var pixel: array<f32, 8>;
    for (var c: u32 = 0u; c < channels; c = c + 1u) {
        let idx = index(c, gid.y, gid.x, width, height);
        pixel[c] = input_image[idx] * params.factors.x;
    }

    if params.factors.y != 1.0 {
        for (var c: u32 = 0u; c < channels; c = c + 1u) {
            let mean = params.means[min(c, 3u)];
            pixel[c] = (pixel[c] - mean) * params.factors.y + mean;
        }
    }

    if channels >= 3u {
        var rgb = vec3<f32>(pixel[0], pixel[1], pixel[2]);
        rgb = apply_saturation(rgb, params.factors.z);
        rgb = apply_hue(rgb, params.factors.w);
        pixel[0] = rgb.x;
        pixel[1] = rgb.y;
        pixel[2] = rgb.z;
    }

    for (var c: u32 = 0u; c < channels; c = c + 1u) {
        let idx = index(c, gid.y, gid.x, width, height);
        output_image[idx] = pixel[c];
    }
}
