struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
    t_min: f32,
    t_max: f32,
};

struct FieldSample {
    sigma: f32,
    radiance: vec3<f32>,
};

struct VolumeUniforms {
    num_rays: u32,
    samples_per_ray: u32,
    delta_scale: f32,
    _padding: u32,
};

@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<storage, read> field_samples: array<FieldSample>;
@group(0) @binding(2) var<storage, read> deltas: array<f32>;
@group(0) @binding(3) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> uniforms: VolumeUniforms;

fn activation_sigma(value: f32) -> f32 {
    return max(value, 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;
    if (ray_index >= uniforms.num_rays) {
        return;
    }

    let _ray = rays[ray_index];
    let samples_per_ray = uniforms.samples_per_ray;
    let base_index = ray_index * samples_per_ray;
    var transmittance = 1.0;
    var rgb = vec3<f32>(0.0, 0.0, 0.0);
    var opacity = 0.0;

    for (var i: u32 = 0u; i < samples_per_ray; i = i + 1u) {
        let sample = field_samples[base_index + i];
        let sigma = activation_sigma(sample.sigma);
        let step = deltas[base_index + i] * uniforms.delta_scale;
        let alpha = 1.0 - exp(-sigma * step);
        let weight = transmittance * alpha;
        rgb += weight * sample.radiance;
        opacity += weight;
        transmittance *= 1.0 - alpha;
    }

    accum[ray_index] = vec4<f32>(rgb, opacity);
}
