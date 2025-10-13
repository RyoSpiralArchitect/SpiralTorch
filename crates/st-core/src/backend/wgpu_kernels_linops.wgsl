// wgpu_kernels_linops.wgsl (v1.8.0) – dot/axpy

struct LParams { n:u32, _pad:u32, _p2:u32, _p3:u32 };
@group(0) @binding(0) var<storage, read>  VX: array<f32>;
@group(0) @binding(1) var<storage, read>  VY: array<f32>;
@group(0) @binding(2) var<storage, read_write> VZ: array<f32>;
@group(0) @binding(3) var<uniform> LP: LParams;

var<workgroup> s_part: array<f32, 256u>;

// y += alpha * x  (alpha is passed via VZ[0], then restored)
@compute @workgroup_size(256)
fn axpy_inplace(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
){
  let alpha = VZ[0];
  var i = gid.x * 256u + lid.x;
  loop {
    if (i >= LP.n) { break; }
    VZ[i] = VY[i] + alpha * VX[i];
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
  loop {
    if (i >= LP.n) { break; }
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
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>
){
  var sum: f32 = 0.0;
  var i = gid.x * 256u + lid.x;
  let total = nwg.x * 256u;
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
