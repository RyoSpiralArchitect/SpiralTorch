
struct RC { nd: u32, n: u32 };
struct RB { c_base: u32, x_base: u32, y_base: u32 };

@group(0) @binding(0) var<storage, read>  C: array<u32>;
@group(0) @binding(1) var<storage, read>  X: array<f32>;
@group(0) @binding(2) var<storage, read>  Y: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;

@group(0) @binding(4) var<storage, read> OUT_SHAPE: array<u32>;
@group(0) @binding(5) var<storage, read> OUT_STRIDES: array<u32>;
@group(0) @binding(6) var<storage, read> C_SHAPE: array<u32>;
@group(0) @binding(7) var<storage, read> C_STRIDES: array<u32>;
@group(0) @binding(8) var<storage, read> X_SHAPE: array<u32>;
@group(0) @binding(9) var<storage, read> X_STRIDES: array<u32>;
@group(0) @binding(10) var<storage, read> Y_SHAPE: array<u32>;
@group(0) @binding(11) var<storage, read> Y_STRIDES: array<u32>;

@group(0) @binding(12) var<uniform> rc: RC;
@group(0) @binding(13) var<uniform> rb: RB;

fn udiv(a:u32, b:u32) -> u32 { return a / b; }
fn umod(a:u32, b:u32) -> u32 { return a % b; }

@compute @workgroup_size(256)
fn where_nd_strided_u8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= rc.n) { return; }
    var rem = i;
    var oc:u32 = rb.c_base; var ox:u32 = rb.x_base; var oy:u32 = rb.y_base;
    for (var d:u32=0u32; d<rc.nd; d=d+1u32) {
        let s = OUT_STRIDES[d];
        let ho = OUT_SHAPE[d];
        let cd = umod(udiv(rem, s), ho);
        rem = umod(rem, s);
        if (C_SHAPE[d] != 1u) { oc = oc + cd * C_STRIDES[d]; }
        if (X_SHAPE[d] != 1u) { ox = ox + cd * X_STRIDES[d]; }
        if (Y_SHAPE[d] != 1u) { oy = oy + cd * Y_STRIDES[d]; }
    }
    let pack = C[oc >> 2u];
    let cbyte = (pack >> ((oc & 3u) * 8u)) & 0xffu;
    let take_x = (cbyte != 0u);
    O[i] = select(Y[oy], X[ox], take_x);
}
