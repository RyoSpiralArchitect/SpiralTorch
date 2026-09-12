struct Shape { len:u32, grid_x:u32, _pad0:u32, _pad1:u32 }
struct Chunk { weights:vec4<f32>, first:u32, count:u32, total:u32, _pad:u32 }
@group(0) @binding(0) var<storage,read> a:array<f32>;
@group(0) @binding(1) var<storage,read> b:array<f32>;
@group(0) @binding(2) var<storage,read> c:array<f32>;
@group(0) @binding(3) var<storage,read> d:array<f32>;
@group(0) @binding(4) var<storage,read_write> output:array<f32>;
@group(0) @binding(5) var<storage,read_write> flags:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> shape:Shape;
@group(0) @binding(7) var<uniform> chunk:Chunk;
CHECKED_ELEMENTWISE

fn source(slot:u32, i:u32) -> f32 {
    switch slot {
        case 0u: { return a[i]; }
        case 1u: { return b[i]; }
        case 2u: { return c[i]; }
        default: { return d[i]; }
    }
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let i = (wid.y*shape.grid_x+wid.x)*256u+lane;
    if (i>=shape.len) { return; }
    var value=0.0;
    var first=0u;
    if (chunk.first==0u) {
        value=checked_apply(OP_MULTIPLY,source(0u,i),chunk.weights[0u]);
        first=1u;
    } else { value=output[i]; }
    for (var slot=first;slot<chunk.count;slot=slot+1u) {
        // Check each product and each ordered sum; cancellation cannot erase overflow.
        let product=checked_apply(OP_MULTIPLY,source(slot,i),chunk.weights[slot]);
        value=checked_apply(OP_ADD,value,product);
    }
    output[i]=value;
}
