@group(0) @binding(0) var<storage, read> upstream: array<u32>;
@group(0) @binding(1) var<storage, read_write> destination: array<atomic<u32>>;

@compute @workgroup_size(1)
fn main() {
    var bits = 0u;
    for (var i = 0u; i < arrayLength(&upstream); i++) { bits |= upstream[i]; }
    atomicStore(&destination[0], select(0u, INVALID_TENSOR_FLAG, bits != 0u));
}
