// crates/st-tensor/src/wgpu_shaders/frac_gl_1d.wgsl
struct Params { n:u32, m:u32, h_alpha:f32; _pad:f32; };

@group(0) @binding(0) var<storage, read>  X : array<f32>;
@group(0) @binding(1) var<storage, read>  W : array<f32>;
@group(0) @binding(2) var<storage, read_write> Y : array<f32>;
@group(0) @binding(3) var<uniform> P : Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) { return; }
    var acc: f32 = 0.0;
    let kmax = min(u32(i), P.m);
    for (var k:u32 = 0u; k <= kmax; k = k + 1u) {
        acc = acc + W[k] * X[i - k];
    }
    Y[i] = acc / P.h_alpha;
}
