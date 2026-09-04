struct Params {
    rows: u32, cols: u32, values: u32, flags: u32,
    scalar: f32, saturation: f32, porosity: f32, pad: f32,
};
@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn gather_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= params.values { return; }
    let row = indices[index / params.cols];
    output[index] = input[row * params.cols + index % params.cols];
}

@compute @workgroup_size(256)
fn scatter_add_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if index >= params.values { return; }
    let row = index / params.cols;
    let col = index % params.cols;
    // Offsets precede token positions. Stable grouping preserves token order.
    var total = 0.0;
    for (var slot = indices[row]; slot < indices[row + 1u]; slot += 1u) {
        let token = indices[params.flags + 1u + slot];
        total += input[token * params.cols + col];
    }
    output[index] = total * params.scalar;
}
