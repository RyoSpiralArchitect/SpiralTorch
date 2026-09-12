struct Shape { len: u32, grid_x: u32, groups: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> upstream: array<u32>;
@group(0) @binding(2) var<uniform> shape: Shape;
@group(1) @binding(0) var<storage, read_write> output: array<f32>;
@group(1) @binding(1) var<storage, read_write> flags: array<atomic<u32>>;

CHECKED_ELEMENTWISE

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group = wid.y * shape.grid_x + wid.x;
    if (group >= shape.groups) { return; }
    let i = group * 256u + lane;
    if (i == 0u) {
        var inherited = 0u;
        for (var j = 0u; j < arrayLength(&upstream); j++) { inherited |= upstream[j]; }
        if (inherited != 0u) { atomicOr(&flags[0], INVALID_TENSOR_FLAG); }
    }
    if (i >= shape.len) { return; }
    let value = source[i];
    check(value);
    output[i] = value;
}
