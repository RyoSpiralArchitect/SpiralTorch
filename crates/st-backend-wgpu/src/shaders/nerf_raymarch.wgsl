// ROUNDED_ADD

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
@group(0) @binding(5) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read> packed_samples: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> field_flags: array<u32>;
@group(0) @binding(8) var<storage, read> width_flags: array<u32>;

struct VolumeState {
    rgb: vec3<f32>,
    rgb_error: vec3<f32>,
    opacity: f32,
    opacity_error: f32,
    depth: f32,
    depth_error: f32,
    valid: bool,
};

fn finite(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn finite3(value: vec3<f32>) -> bool {
    return finite(value.x) && finite(value.y) && finite(value.z);
}

fn initial_state() -> VolumeState {
    return VolumeState(vec3<f32>(0.0), vec3<f32>(0.0), 0.0, 0.0, 0.0, 0.0, true);
}

fn compensated_add(sum: f32, error: f32, value: f32) -> vec2<f32> {
    let increment = rounded_add(value, -error);
    let total = rounded_add(sum, increment);
    let residual = rounded_add(rounded_add(total, -sum), -increment);
    return vec2<f32>(total, residual);
}

fn integrate_step(previous: VolumeState, sample: FieldSample, width: f32) -> VolumeState {
    var state = previous;
    if !state.valid {
        return state;
    }
    if !finite(sample.sigma) || !finite3(sample.radiance) || !finite(width) || width < 0.0 {
        state.valid = false;
        return state;
    }
    let tau = max(sample.sigma, 0.0) * width;
    if !finite(tau) {
        state.valid = false;
        return state;
    }
    var alpha = 1.0 - exp(-tau);
    if tau < 0.01 {
        // expm1 is not a WGSL builtin. This expansion retains thin opacity.
        alpha = tau * (1.0 + tau * (-0.5 + tau * (1.0 / 6.0 - tau / 24.0)));
    }
    // A compensated optical-depth prefix avoids repeated f32 attenuation drift
    // and preserves opaque tails even when alpha rounds to exactly one.
    let weight = exp(-state.depth) * alpha;
    for (var channel = 0u; channel < 3u; channel += 1u) {
        let value = compensated_add(state.rgb[channel], state.rgb_error[channel], weight * sample.radiance[channel]);
        state.rgb[channel] = value.x;
        state.rgb_error[channel] = value.y;
    }
    let opacity = compensated_add(state.opacity, state.opacity_error, weight);
    state.opacity = opacity.x;
    state.opacity_error = opacity.y;
    let depth = compensated_add(state.depth, state.depth_error, tau);
    state.depth = depth.x;
    state.depth_error = depth.y;
    state.valid = finite3(state.rgb) && finite3(state.rgb_error)
        && finite(state.opacity) && finite(state.opacity_error)
        && finite(state.depth) && finite(state.depth_error);
    return state;
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
    var state = initial_state();

    for (var i: u32 = 0u; i < samples_per_ray; i = i + 1u) {
        let sample = field_samples[base_index + i];
        let step = deltas[base_index + i] * uniforms.delta_scale;
        state = integrate_step(state, sample, step);
    }

    // Legacy callers have no guard binding: never hide invalid data behind a
    // finite partial integral. Keep failure observable in the output itself.
    if state.valid {
        accum[ray_index] = vec4<f32>(state.rgb, state.opacity);
    } else {
        // WGSL const-expressions cannot produce NaN. The ray payload keeps
        // this a runtime bitcast while preserving a quiet-NaN exponent.
        accum[ray_index] = vec4<f32>(bitcast<f32>(0x7fc00000u | (ray_index & 0x003fffffu)));
    }
}

// ResidentTensor's [sigma, r, g, b] rows need no padded FieldSample copy.
@compute @workgroup_size(64)
fn resident_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let ray_index = global_id.x;
    if ray_index >= uniforms.num_rays {
        return;
    }
    if ray_index == 0u {
        atomicOr(&flags[0], field_flags[0] | width_flags[0]);
    }
    let base_index = ray_index * uniforms.samples_per_ray;
    var state = initial_state();
    for (var i = 0u; i < uniforms.samples_per_ray; i += 1u) {
        let sample = packed_samples[base_index + i];
        state = integrate_step(state, FieldSample(sample.x, sample.yzw), deltas[base_index + i]);
    }
    accum[ray_index] = vec4<f32>(state.rgb, state.opacity);
    if !state.valid {
        atomicOr(&flags[0], 0x80000000u);
    }
}
