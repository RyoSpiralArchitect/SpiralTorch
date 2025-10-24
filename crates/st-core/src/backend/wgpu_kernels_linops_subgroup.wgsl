// wgpu_kernels_linops_subgroup.wgsl (v1.9.1) — vector primitives w/ subgroup reductions
//
// Mirrors the baseline linops kernels but uses subgroup intrinsics for the heavy
// reductions. This keeps compatibility with the portable path while letting
// subgroup-capable GPUs (M-series, RDNA, Xe, etc.) execute warp-synchronous
// reductions without bouncing through shared memory.

enable subgroups;

struct LParams {
  dims:    vec4<u32>;  // x: length, y: partial_count, z: unused, w: unused
  scalars: vec4<f32>;  // x: alpha (scale/axpy), others reserved
};

@group(0) @binding(0) var<storage, read>  VX: array<f32>;
@group(0) @binding(1) var<storage, read>  VY: array<f32>;
@group(0) @binding(2) var<storage, read_write> VZ: array<f32>;
@group(0) @binding(3) var<uniform> LP: LParams;

var<workgroup> s_part: array<f32, 256u>;

fn vec_len()->u32 { return LP.dims.x; }
fn vec_partial_count()->u32 { return LP.dims.y; }

@compute @workgroup_size(256)
fn dot_partials_subgroup(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) lindex: u32,
  @builtin(subgroup_invocation_id) sg_lane: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  var sum: f32 = 0.0;
  var i = wid.x * 256u + lid.x;
  let n = vec_len();
  loop {
    if (i >= n) { break; }
    sum = sum + VX[i] * VY[i];
    i += 256u;
  }

  let sg_sum = subgroupAdd(sum);
  if (sg_lane == 0u) {
    s_part[sg_id] = sg_sum;
  }
  workgroupBarrier();

  if (lindex == 0u) {
    var total: f32 = 0.0;
    var sg: u32 = 0u;
    let num_sg = (256u + sg_size - 1u) / max(sg_size, 1u);
    loop {
      if (sg >= num_sg) { break; }
      total = total + s_part[sg];
      sg = sg + 1u;
    }
    VZ[wid.x] = total;
  }
}

@compute @workgroup_size(256)
fn dot_finalize_subgroup(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(local_invocation_index) lindex: u32,
  @builtin(subgroup_invocation_id) sg_lane: u32,
  @builtin(subgroup_id) sg_id: u32,
  @builtin(subgroup_size) sg_size: u32,
) {
  var sum: f32 = 0.0;
  var i = gid.x * 256u + lid.x;
  let total = vec_partial_count();
  loop {
    if (i >= total) { break; }
    sum = sum + VZ[i];
    i += 256u;
  }

  let sg_sum = subgroupAdd(sum);
  if (sg_lane == 0u) {
    s_part[sg_id] = sg_sum;
  }
  workgroupBarrier();

  if (lindex == 0u) {
    var total_sum: f32 = 0.0;
    var sg: u32 = 0u;
    let num_sg = (256u + sg_size - 1u) / max(sg_size, 1u);
    loop {
      if (sg >= num_sg) { break; }
      total_sum = total_sum + s_part[sg];
      sg = sg + 1u;
    }
    VZ[0] = total_sum;
  }
}
