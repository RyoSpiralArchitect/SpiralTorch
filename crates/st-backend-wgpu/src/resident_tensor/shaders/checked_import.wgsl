struct Params {
    len: u32,
    groups_x: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let index = (group.y * params.groups_x + group.x) * 256u + lane;
    if index >= params.len { return; }
    if (bitcast<u32>(values[index]) & 0x7f800000u) == 0x7f800000u {
        atomicOr(&flags[0], 0x80000000u);
    }
}
