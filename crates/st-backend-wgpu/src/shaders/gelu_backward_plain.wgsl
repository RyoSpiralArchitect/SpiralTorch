struct Params { len: u32, groups_x: u32, groups: u32, padding: u32 };
@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read> upstream: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// GELU_DERIVATIVE

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group = wid.y * params.groups_x + wid.x;
    if (group >= params.groups) { return; }
    let index = group * 256u + lane;
    if (index < params.len) { gradient[index] = upstream[index] * gelu_prime(z[index]); }
}
