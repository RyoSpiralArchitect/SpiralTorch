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
    seed: u32,
};

struct SamplePoint {
    position: vec3<f32>,
    distance: f32,
};

@group(0) @binding(0) var<storage, read> rays: array<Ray>;
@group(0) @binding(1) var<storage, read_write> samples: array<SamplePoint>;
@group(0) @binding(2) var<storage, read_write> deltas: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: SampleUniforms;
@group(0) @binding(4) var<storage, read_write> flags: array<atomic<u32>>;

fn generate_offset(base: u32, sample: u32) -> f32 {
    var bits = uniforms.seed ^ (base * 0x9e3779b9u + sample * 0x85ebca6bu);
    bits = (bits ^ (bits >> 16u)) * 0x7feb352du;
    bits = (bits ^ (bits >> 15u)) * 0x846ca68bu;
    bits ^= bits >> 16u;
    return f32(bits >> 8u) * (1.0 / 16777216.0);
}

fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn sample_ray(ray_index: u32) -> bool {
    let ray = rays[ray_index];
    let samples_per_ray = uniforms.samples_per_ray;
    if samples_per_ray == 0u || !finite(ray.t_min) || !finite(ray.t_max)
        || ray.t_max < ray.t_min {
        return false;
    }
    let base_index = ray_index * samples_per_ray;
    let span = ray.t_max - ray.t_min;
    let interval = span / f32(samples_per_ray);
    if !finite(span) || !finite(interval) || (span > 0.0 && interval == 0.0) {
        return false;
    }

    for (var i: u32 = 0u; i < samples_per_ray; i = i + 1u) {
        var offset = 0.5;
        if uniforms.stratified != 0u {
            offset = generate_offset(ray_index, i);
        }
        let t = ray.t_min + (f32(i) + offset) * interval;
        let sample_index = base_index + i;
        let position = ray.origin + ray.direction * t;
        if !finite(t) || !finite(position.x) || !finite(position.y) || !finite(position.z) {
            return false;
        }
        samples[sample_index] = SamplePoint(position, t);
        // Jitter selects a point; it must not shrink the bin's integration width.
        deltas[sample_index] = interval;
    }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < uniforms.num_rays {
        let valid = sample_ray(global_id.x);
    }
}

@compute @workgroup_size(64)
fn checked_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x < uniforms.num_rays && !sample_ray(global_id.x) {
        atomicOr(&flags[0], 0x80000000u);
    }
}
