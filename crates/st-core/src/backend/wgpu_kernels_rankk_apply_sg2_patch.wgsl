// wgpu_kernels_rankk.wgsl (v1.8.5 additions)
// Only the changed apply_sg2 section is shown here for brevity. Other entries remain as in v1.8.3.

var<workgroup> sg_totals: array<u32, 8u>;
var<workgroup> sg_bases:  array<u32, 8u>;
var<workgroup> sg_temp:   array<u32, 8u>;

@compute @workgroup_size(256)
fn midk_compact_apply_sg2(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(subgroup_size)        sg_size: u32,
  @builtin(subgroup_invocation_id) sg_lane: u32
){
  let r = wid.y;
  let tile = gid.x;
  if (r>=CP.rows || tile>=CP.tiles_x) { return; }

  let start = tile * 256u;
  let base  = PREFIX[r*CP.tiles_x + tile];

  // per-lane flag
  var flag: u32 = 0u;
  let col = start + lid.x;
  if (col < CP.cols) { if (CMASK[r*CP.row_stride + col] != 0u) { flag = 1u; } }

  let sgsz = max(sg_size, 1u);
  let sgc  = 256u / sgsz;
  let sg_id = lid.x / sgsz;

  // subgroup local exclusive + totals (portable path)
  var local_excl: u32 = 0u;
  var sg_total:  u32 = 0u;
  let sg_start = start + sg_id * sgsz;
  for (var j:u32=0u; j<sgsz; j=j+1u) {
    let idx = sg_start + j;
    if (idx < CP.cols) {
      let f = u32(CMASK[r*CP.row_stride + idx] != 0u);
      if (j < sg_lane) { local_excl = local_excl + f; }
      sg_total = sg_total + f;
    }
  }

  if (sg_lane == 0u) { sg_totals[sg_id] = sg_total; }
  workgroupBarrier();

  // --- New: multi-leader parallel prefix across subgroup totals (Blelloch over <=8 values) ---
  // First sgc lanes (lid.x < sgc) run a tiny workgroup scan; others idle. Two global fences total.
  if (lid.x < sgc) { sg_temp[lid.x] = sg_totals[lid.x]; }
  workgroupBarrier();

  // upsweep
  var d = 1u; var n = sgc;
  loop {
    if (d >= n) { break; }
    let i = ( (lid.x + 1u) * (d<<1u) ) - 1u;
    if (lid.x < n && i < n) { sg_temp[i] = sg_temp[i] + sg_temp[i - d]; }
    d = d << 1u;
    workgroupBarrier();
  }

  // convert to exclusive
  if (lid.x == 0u) { sg_temp[n-1u] = 0u; }
  workgroupBarrier();

  // downsweep
  var step = n >> 1u;
  var stride = step;
  loop {
    if (step == 0u) { break; }
    let i = ( (lid.x + 1u) * (stride<<1u) ) - 1u;
    if (lid.x < n && i < n) {
      let t = sg_temp[i - stride];
      sg_temp[i - stride] = sg_temp[i];
      sg_temp[i] = sg_temp[i] + t;
    }
    step = step >> 1u; stride = max(stride >> 1u, 1u);
    workgroupBarrier();
  }

  if (lid.x < sgc) { sg_bases[lid.x] = sg_temp[lid.x]; }
  workgroupBarrier();
  let sg_base = sg_bases[sg_id];

  // write compacted
  if (flag == 1u && col < CP.cols) {
    let pos = base + sg_base + local_excl;
    OUTVAL[r*CP.cols + pos] = CX[r*CP.row_stride + col];
  }
}
