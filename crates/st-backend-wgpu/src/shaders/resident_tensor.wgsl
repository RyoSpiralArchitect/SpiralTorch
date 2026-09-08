// Layout metadata and operation ids are supplied by the Rust semantic core.
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;
@group(0) @binding(4) var<storage, read> a_flags: array<u32>;
@group(0) @binding(5) var<storage, read> b_flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> flags: array<atomic<u32>>;

fn check(x: f32) {
    if ((bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&flags[0], INVALID_TENSOR_FLAG);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group = wid.y * params[3] + wid.x;
    if (group >= params[4]) { return; }
    let i = group * 256u + lane;
    if (i == 0u) {
        var inherited = 0u;
        for (var j = 0u; j < params[7]; j++) { inherited |= a_flags[j]; }
        for (var j = 0u; j < params[8]; j++) { inherited |= b_flags[j]; }
        if (inherited != 0u) { atomicOr(&flags[0], INVALID_TENSOR_FLAG); }
    }
    if (i >= params[0]) { return; }
    let rank = params[1];
    var logical = i;
    var ai = params[5];
    var bi = params[6];
    for (var axis = rank; axis > 0u; axis--) {
        let d = axis - 1u;
        let coordinate = logical % params[9u + d];
        logical /= params[9u + d];
        ai += coordinate * params[9u + rank + d];
        bi += coordinate * params[9u + 2u * rank + d];
    }
    let x = a[ai];
    check(x);
    var value = x;
    switch params[2] {
        case OP_ADD: { let y = b[bi]; check(y); value = x + y; }
        case OP_MULTIPLY: { let y = b[bi]; check(y); value = x * y; }
        case OP_RELU: { value = max(x, 0.0); }
        case OP_GELU: {
            let square = x * x;
            let cubic = square * x;
            let inner = 0.7978846 * (x + 0.044715 * cubic);
            check(square); check(cubic); check(inner);
            value = 0.5 * x * (1.0 + tanh(clamp(inner, -10.0, 10.0)));
        }
        default: {}
    }
    check(value);
    out[i] = value;
}
