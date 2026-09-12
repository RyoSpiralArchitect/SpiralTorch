struct Chunk { weights:vec4<f32>, first:u32, count:u32, total:u32, _pad:u32 }
@group(0) @binding(0) var<storage,read> sources:array<u32>;
@group(0) @binding(1) var<storage,read_write> flags:array<atomic<u32>>;
@group(0) @binding(2) var<uniform> chunk:Chunk;
@compute @workgroup_size(1)
fn main() {
    var inherited=0u;
    for (var i=0u;i<chunk.total;i=i+1u) { inherited=inherited|sources[i]; }
    atomicOr(&flags[0],inherited);
}
