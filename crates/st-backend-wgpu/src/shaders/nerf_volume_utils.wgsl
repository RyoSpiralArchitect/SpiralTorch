struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    t_min: f32,
    t_max: f32,
};

struct SampleUniforms {
    num_rays: u32,
    samples_per_ray: u32,
    stratified: u32,
    _padding: u32,
};

struct SamplePoint {
    position: vec3<f32>,
    distance: f32,
};

@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<storage, read_write> samples: array<SamplePoint>;
@group(0) @binding(2) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: SampleUniforms;

fn generate_offset(base: u32, sample: u32) -> f32 {
    let step = 1103515245u * (base + sample) + 12345u;
    let bits = (step >> 9u) & 0x007fffffu;
    return f32(bits) / f32(0x007fffff);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;
    if (ray_index >= uniforms.num_rays) {
        return;
    }

    let ray = rays[ray_index];
    let samples_per_ray = uniforms.samples_per_ray;
    let base_index = ray_index * samples_per_ray;
    let span = max(ray.t_max - ray.t_min, 1e-3);
    let interval = span / f32(samples_per_ray);

    for (var i: u32 = 0u; i < samples_per_ray; i = i + 1u) {
        let offset = if uniforms.stratified == 0u {
            0.5
        } else {
            generate_offset(ray_index, i)
        };
        let clamped = clamp(offset, 0.0, 0.999);
        let t = ray.t_min + (f32(i) + clamped) * interval;
        let next_t = if i + 1u == samples_per_ray {
            ray.t_max
        } else {
            ray.t_min + f32(i + 1u) * interval
        };
        let delta = max(next_t - t, 1e-3);
        let sample_index = base_index + i;
        let position = ray.origin + ray.direction * t;
        samples[sample_index] = SamplePoint(position, t);
        deltas[sample_index] = delta;
    }
}
