
// Unified WGSL kernels (subset sufficient)
struct U32One { n: u32; };

@group(0) @binding(0) var<storage, read>  add_a: array<f32>;
@group(0) @binding(1) var<storage, read>  add_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> add_c: array<f32>;
@group(0) @binding(3) var<uniform> addN: U32One;

@compute @workgroup_size(256)
fn add_vec(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= addN.n) { return; }
  add_c[i] = add_a[i] + add_b[i];
}

// Reduce 1/2-pass (rows, cols) simplified
struct NdWGInfo {
  n_rows: u32, n_cols: u32, kdims: u32, rdims: u32,
  kshape: vec4<u32>, rshape: vec4<u32>, kstride: vec4<i32>, rstride: vec4<i32>,
}
var<workgroup> sdata: array<f32,256u>;

@group(2) @binding(0) var<storage, read>   rx_buf: array<f32>;
@group(2) @binding(1) var<storage, read_write> rout_buf: array<f32>;
@group(2) @binding(2) var<uniform> rinfo: NdWGInfo;

@compute @workgroup_size(256)
fn reduce_nd_wg_sum(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  if (wid.x >= rinfo.n_rows) { return; }
  var acc: f32 = 0.0; var c:u32 = lid.x;
  while (c < rinfo.n_cols) { acc += rx_buf[wid.x * rinfo.n_cols + c]; c += 256u; }
  sdata[lid.x] = acc; workgroupBarrier();
  var s:u32 = 128u;
  loop{
    if (lid.x < s) { sdata[lid.x] += sdata[lid.x+s]; } workgroupBarrier();
    if (s==1u) { break; } s = s>>1u;
  }
  if (lid.x==0u) { rout_buf[wid.x] = sdata[0u]; }
}

// Transpose 2D
struct Trans2DInfo { rows:u32, cols:u32, stride_x:u32, stride_y:u32 };
@group(6) @binding(0) var<storage, read>  t2_x: array<f32>;
@group(6) @binding(1) var<storage, read_write> t2_y: array<f32>;
@group(6) @binding(2) var<uniform> t2: Trans2DInfo;
@compute @workgroup_size(256)
fn transpose_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tot = t2.rows * t2.cols; let i = gid.x; if (i>=tot) { return; }
  let r = i / t2.cols; let c = i % t2.cols;
  let ox = r * t2.cols + c; let oy = c * t2.rows + r;
  t2_y[oy] = t2_x[ox];
}

// Tiled GEMM 16x16
struct MatmulInfo { m:u32, n:u32, k:u32, tile:u32, lda:u32, ldb:u32, ldc:u32, batch:u32, stride_a:u32, stride_b:u32, stride_c:u32, _p:u32 };
@group(5) @binding(0) var<storage, read>  mm_a: array<f32>;
@group(5) @binding(1) var<storage, read>  mm_b: array<f32>;
@group(5) @binding(2) var<storage, read_write> mm_c: array<f32>;
@group(5) @binding(3) var<uniform> mm: MatmulInfo;
var<workgroup> Asub: array<array<f32,16>,16>;
var<workgroup> Bsub: array<array<f32,16>,16>;
@compute @workgroup_size(16,16,1)
fn matmul_tiled(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let TILE:u32=16u; let row = wg.y*TILE + li.y; let col = wg.x*TILE + li.x;
  var acc:f32 = 0.0; let tiles = (mm.k + TILE - 1u)/TILE;
  for (var t:u32=0u; t<tiles; t=t+1u) {
    let ac = t*TILE + li.x; let br = t*TILE + li.y;
    Asub[li.y][li.x] = (row<mm.m && ac<mm.k) ? mm_a[row*mm.lda + ac] : 0.0;
    Bsub[li.y][li.x] = (br<mm.k && col<mm.n) ? mm_b[br*mm.ldb + col] : 0.0;
    workgroupBarrier();
    for (var j:u32=0u; j<TILE; j=j+1u) { acc += Asub[li.y][j]*Bsub[j][li.x]; }
    workgroupBarrier();
  }
  if (row<mm.m && col<mm.n) { mm_c[row*mm.ldc + col] = acc; }
}
