struct RC { nd: u32, n: u32 };
struct RB { c_base: u32, x_base: u32, y_base: u32 };

@group(0) @binding(0)  var<storage, read>  C: array<u32>;
@group(0) @binding(1)  var<storage, read>  X: array<f32>;
@group(0) @binding(2)  var<storage, read>  Y: array<f32>;
@group(0) @binding(3)  var<storage, read_write> O: array<f32>;
@group(0) @binding(4)  var<storage, read>  OUT_SHAPE: array<u32>;
@group(0) @binding(5)  var<storage, read>  OUT_STRIDES: array<u32>;
@group(0) @binding(6)  var<storage, read>  C_SHAPE: array<u32>;
@group(0) @binding(7)  var<storage, read>  C_STRIDES: array<u32>;
@group(0) @binding(8)  var<storage, read>  X_SHAPE: array<u32>;
@group(0) @binding(9)  var<storage, read>  X_STRIDES: array<u32>;
@group(0) @binding(10) var<storage, read>  Y_SHAPE: array<u32>;
@group(0) @binding(11) var<storage, read>  Y_STRIDES: array<u32>;
@group(0) @binding(12) var<uniform>        uRC: RC;
@group(0) @binding(13) var<uniform>        uRB: RB;

@compute @workgroup_size(256)
fn where_nd_strided_u8(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= uRC.n) { return; }
  var rem = i;
  var oc = uRB.c_base;
  var ox = uRB.x_base;
  var oy = uRB.y_base;
  for (var d:u32=0u; d<uRC.nd; d=d+1u) {
    let s  = OUT_STRIDES[d];
    let ho = OUT_SHAPE[d];
    let cd = (rem / s) % ho;
    rem = rem % s;
    if (C_SHAPE[d] != 1u) { oc += cd * C_STRIDES[d]; }
    if (X_SHAPE[d] != 1u) { ox += cd * X_STRIDES[d]; }
    if (Y_SHAPE[d] != 1u) { oy += cd * Y_STRIDES[d]; }
  }
  let cw = C[oc >> 2u];
  let shift = (oc & 3u) * 8u;
  let cb = (cw >> shift) & 0xffu;
  let vx = X[ox];
  let vy = Y[oy];
  O[i] = select(vy, vx, cb != 0u);
}
