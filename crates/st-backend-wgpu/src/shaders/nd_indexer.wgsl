// Compute flattened indices for ND tensors with arbitrary strides and segment offsets.
// Each invocation handles one logical element and emits both the global pointer and the
// segment id that can be consumed by follow-up kernels.

struct Params {
  dims: vec4<u32>,
  strides: vec4<u32>,
  segments: u32,
  segment_stride: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> positions: array<u32>;

@group(0) @binding(2)
var<storage, write> out_indices: array<u32>;

@group(0) @binding(3)
var<storage, write> out_segments: array<u32>;

fn decode_linear(id: u32, dims: vec4<u32>) -> vec4<u32> {
  var coord = vec4<u32>(0u);
  var remaining = id;
  for (var i: i32 = 3; i >= 0; i = i - 1) {
    let dim = dims[i];
    if (dim == 0u) {
      continue;
    }
    coord[i] = remaining % dim;
    remaining = remaining / dim;
  }
  return coord;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= arrayLength(&positions)) {
    return;
  }
  let logical = positions[idx];
  let coord = decode_linear(logical, params.dims);
  let flat = dot(vec4<u32>(coord), params.strides);
  let seg = (flat / params.segment_stride) % params.segments;
  out_indices[idx] = flat;
  out_segments[idx] = seg;
}
