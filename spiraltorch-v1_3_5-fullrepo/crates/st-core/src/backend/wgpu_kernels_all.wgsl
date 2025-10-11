
// SpiralTorch WGPU unified kernels (subset sufficient for v1.3.5 demo)

// -- add_vec
struct U32One { n: u32; };
@group(0) @binding(0) var<storage, read> a0: array<f32>;
@group(0) @binding(1) var<storage, read> b0: array<f32>;
@group(0) @binding(2) var<storage, read_write> c0: array<f32>;
@group(0) @binding(3) var<uniform> u0: U32One;
@compute @workgroup_size(256)
fn add_vec(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u0.n) { return; }
  c0[i] = a0[i] + b0[i];
}

// -- transpose_2d
struct T2 { rows: u32, cols: u32, stride_x: u32, stride_y: u32 };
@group(6) @binding(0) var<storage, read>  t2x: array<f32>;
@group(6) @binding(1) var<storage, read_write> t2y: array<f32>;
@group(6) @binding(2) var<uniform> t2: T2;
@compute @workgroup_size(256)
fn transpose_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tot = t2.rows * t2.cols;
  let i = gid.x; if (i >= tot) { return; }
  let r = i / t2.cols; let c = i % t2.cols;
  let ox = r * t2.cols + c;
  let oy = c * t2.rows + r;
  t2y[oy] = t2x[ox];
}

// -- transpose_2d_batched
struct TB { rows: u32, cols: u32, batches: u32, _p: u32, stride_x: u32, stride_y: u32, _p1:u32, _p2:u32 };
@group(7) @binding(0) var<storage, read>  tbx: array<f32>;
@group(7) @binding(1) var<storage, read_write> tby: array<f32>;
@group(7) @binding(2) var<uniform> tb: TB;
@compute @workgroup_size(256)
fn transpose_2d_batched(@builtin(global_invocation_id) gid: vec3<u32>) {
  let per = tb.rows * tb.cols;
  let tot = per * tb.batches;
  let i = gid.x; if (i >= tot) { return; }
  let b = i / per; let l = i % per;
  let r = l / tb.cols; let c = l % tb.cols;
  let ox = b * tb.stride_x + r * tb.cols + c;
  let oy = b * tb.stride_y + c * tb.rows + r;
  tby[oy] = tbx[ox];
}

// -- softmax bwd: rowwise dot + fused
@group(8) @binding(0) var<storage, read>  go8: array<f32>;
@group(8) @binding(1) var<storage, read>  y8: array<f32>;
@group(8) @binding(2) var<storage, read_write> dot8: array<f32>;
@group(8) @binding(3) var<uniform> rc8: vec2<u32>;
var<workgroup> s8: array<f32,256u>;
@compute @workgroup_size(256)
fn rowwise_dot_gy_wg(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let R = rc8.x; let C = rc8.y;
  if (wid.x >= R) { return; }
  var acc: f32 = 0.0; var j: u32 = lid.x;
  while (j < C) { acc = acc + go8[wid.x*C+j]*y8[wid.x*C+j]; j = j + 256u; }
  s8[lid.x] = acc; workgroupBarrier();
  var stride:u32=128u; loop{
    if (lid.x < stride){ s8[lid.x]+=s8[lid.x+stride]; } workgroupBarrier();
    if (stride==1u){break;} stride=stride>>1u;
  }
  if (lid.x==0u){ dot8[wid.x]=s8[0u]; }
}
@group(9) @binding(0) var<storage, read>  go9: array<f32>;
@group(9) @binding(1) var<storage, read>  y9: array<f32>;
@group(9) @binding(2) var<storage, read>  dot9: array<f32>;
@group(9) @binding(3) var<storage, read_write> gx9: array<f32>;
@group(9) @binding(4) var<uniform> rc9: vec2<u32>;
@compute @workgroup_size(256)
fn softmax_bw_from_dot(@builtin(global_invocation_id) gid: vec3<u32>) {
  let R = rc9.x; let C = rc9.y; let i = gid.x; if (i >= R*C){return;}
  let r = i / C; gx9[i] = (go9[i] - dot9[r]) * y9[i];
}

// -- tiled GEMM (2D/batched, TILE=16)
struct MM { m:u32, n:u32, k:u32, tile:u32, lda:u32, ldb:u32, ldc:u32, batch:u32, stride_a:u32, stride_b:u32, stride_c:u32, _p:u32 };
@group(5) @binding(0) var<storage, read>  mma: array<f32>;
@group(5) @binding(1) var<storage, read>  mmb: array<f32>;
@group(5) @binding(2) var<storage, read_write> mmc: array<f32>;
@group(5) @binding(3) var<uniform> mm: MM;
var<workgroup> Asub: array<array<f32,16>,16>;
var<workgroup> Bsub: array<array<f32,16>,16>;
@compute @workgroup_size(16,16,1)
fn matmul_tiled(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let TILE:u32=16u; let row = wg.y*TILE+lid.y; let col = wg.x*TILE+lid.x;
  var acc:f32=0.0; let tiles=(mm.k+TILE-1u)/TILE;
  for (var t:u32=0u; t<tiles; t=t+1u) {
    let ac = t*TILE + lid.x; let br = t*TILE + lid.y;
    Asub[lid.y][lid.x] = (row<mm.m && ac<mm.k) ? mma[row*mm.lda + ac] : 0.0;
    Bsub[lid.y][lid.x] = (br<mm.k && col<mm.n) ? mmb[br*mm.ldb + col] : 0.0;
    workgroupBarrier();
    for (var j:u32=0u; j<TILE; j=j+1u) { acc += Asub[lid.y][j]*Bsub[j][lid.x]; }
    workgroupBarrier();
  }
  if (row<mm.m && col<mm.n) { mmc[row*mm.ldc+col] = acc; }
}
@compute @workgroup_size(16,16,1)
fn matmul_tiled_batched(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let TILE:u32=16u; let row = wg.y*TILE+lid.y; let col = wg.x*TILE+lid.x; let b = wg.z;
  var acc:f32=0.0; let tiles=(mm.k+TILE-1u)/TILE;
  let ba = b*mm.stride_a; let bb = b*mm.stride_b; let bc = b*mm.stride_c;
  for (var t:u32=0u; t<tiles; t=t+1u) {
    let ac = t*TILE + lid.x; let br = t*TILE + lid.y;
    Asub[lid.y][lid.x] = (row<mm.m && ac<mm.k) ? mma[ba + row*mm.lda + ac] : 0.0;
    Bsub[lid.y][lid.x] = (br<mm.k && col<mm.n) ? mmb[bb + br*mm.ldb + col] : 0.0;
    workgroupBarrier();
    for (var j:u32=0u; j<TILE; j=j+1u) { acc += Asub[lid.y][j]*Bsub[j][lid.x]; }
    workgroupBarrier();
  }
  if (row<mm.m && col<mm.n) { mmc[bc + row*mm.ldc+col] = acc; }
}
