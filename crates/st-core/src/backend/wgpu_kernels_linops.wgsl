// wgpu_kernels_linops.wgsl (v1.9.0) – vector primitives (copy/scale/axpy) + dot reduction

struct LParams {
  dims:    vec4<u32>,  // x: length, y: partial_count, z: unused, w: unused
  scalars: vec4<f32>,  // x: alpha (scale/axpy), others reserved
};
@group(0) @binding(0) var<storage, read>  VX: array<f32>;
@group(0) @binding(1) var<storage, read>  VY: array<f32>;
@group(0) @binding(2) var<storage, read_write> VZ: array<f32>;
@group(0) @binding(3) var<uniform> LP: LParams;

var<workgroup> s_part: array<f32, 256u>;

fn vec_len()->u32 { return LP.dims.x; }
fn vec_partial_count()->u32 { return LP.dims.y; }
fn alpha()->f32 { return LP.scalars.x; }

// z = x
@compute @workgroup_size(256)
fn copy_vec(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
){
  var i = gid.x * 256u + lid.x;
  let n = vec_len();
  loop {
    if (i >= n) { break; }
    VZ[i] = VX[i];
    i += 256u;
  }
}

// z = beta (beta stored in scalars.y)
@compute @workgroup_size(256)
fn fill_vec(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
){
  var i = gid.x * 256u + lid.x;
  let n = vec_len();
  let value = LP.scalars.y;
  loop {
    if (i >= n) { break; }
    VZ[i] = value;
    i += 256u;
  }
}

// z = alpha * y
@compute @workgroup_size(256)
fn scale_inplace(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
){
  var i = gid.x * 256u + lid.x;
  let n = vec_len();
  let s = alpha();
  loop {
    if (i >= n) { break; }
    VZ[i] = VY[i] * s;
    i += 256u;
  }
}

// z = y + alpha * x (can be in-place if VZ aliases VY)
@compute @workgroup_size(256)
fn axpy_inplace(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
){
  let s = alpha();
  var i = gid.x * 256u + lid.x;
  let n = vec_len();
  loop {
    if (i >= n) { break; }
    VZ[i] = VY[i] + s * VX[i];
    i += 256u;
  }
}

// partial dot: OUT partials in VZ (same length as grid)
// Assumes grid_x = ceil(n/256)
@compute @workgroup_size(256)
fn dot_partials(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
){
  var sum: f32 = 0.0;
  var i = wid.x * 256u + lid.x;
  let n = vec_len();
  loop {
    if (i >= n) { break; }
    sum = sum + VX[i]*VY[i];
    i += 256u;
  }
  s_part[lid.x] = sum; workgroupBarrier();

  // reduce
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) {
      s_part[lid.x] = s_part[lid.x] + s_part[lid.x + stride];
    }
    stride = stride >> 1u;
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    VZ[wid.x] = s_part[0];
  }
}

// finalize: reduce partials in VZ into VZ[0]
@compute @workgroup_size(256)
fn dot_finalize(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
){
  var sum: f32 = 0.0;
  var i = gid.x * 256u + lid.x;
  let total = vec_partial_count();
  loop {
    if (i >= total) { break; }
    sum = sum + VZ[i];
    i += 256u;
  }
  s_part[lid.x] = sum; workgroupBarrier();
  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { s_part[lid.x] = s_part[lid.x] + s_part[lid.x+stride]; }
    stride >>= 1u; workgroupBarrier();
  }
  if (lid.x == 0u) { VZ[0] = s_part[0]; }
}
