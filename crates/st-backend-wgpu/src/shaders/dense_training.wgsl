// Shared native/browser training operations. Matrix VJPs use the canonical GEMM.
struct Params {
    rows: u32, cols: u32, len: u32, stage: u32,
    stages: u32, gelu: u32, groups_x: u32, partials: u32,
};
struct Step { rate: f32, _pad0: f32, _pad1: f32, _pad2: f32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read> d: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read_write> aux: array<f32>;
@group(0) @binding(6) var<storage, read_write> validation: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p: Params;
@group(0) @binding(8) var<uniform> step: Step;
var<workgroup> sums: array<f32, 256>;

fn check(value: f32, flag: u32) {
    if ((bitcast<u32>(value) & 0x7f800000u) == 0x7f800000u) {
        atomicOr(&validation[p.stage], flag);
    }
}
fn group_index(wid: vec3<u32>) -> u32 { return wid.y * p.groups_x + wid.x; }
fn index(wid: vec3<u32>, lane: u32) -> u32 { return group_index(wid) * 256u + lane; }

// Same saturated tanh derivative as st_tensor::gelu_derivative.
fn gelu_prime(x: f32) -> f32 {
    if (abs(x) >= 10.0) { return select(0.0, 1.0, x > 0.0); }
    let square = x * x;
    let inner = 0.7978846 * (x + 0.044715 * x * square);
    let t = tanh(clamp(inner, -10.0, 10.0));
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t*t) * 0.7978846 * (1.0 + 3.0 * 0.044715 * square);
}

@compute @workgroup_size(256)
fn delta(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i < p.len) {
        var derivative = 1.0;
        if (p.gelu != 0u) { derivative = gelu_prime(a[i]); }
        let value = b[i] * derivative;
        check(derivative, 1024u); check(value, 1024u);
        out[i] = value;
    }
}

@compute @workgroup_size(256)
fn bias_gradient(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let col = index(wid, lane);
    if (col < p.cols) {
        var value = 0.0;
        for (var row = 0u; row < p.rows; row = row + 1u) {
            value = value + a[row * p.cols + col];
            check(value, 2048u);
        }
        out[col] = value;
    }
}

@compute @workgroup_size(256)
fn mse_partials(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    var value = 0.0;
    if (i < p.len) {
        let diff = a[i] - b[i];
        let square = diff * diff;
        let seed = diff * (2.0 / f32(p.len));
        check(diff, 1u); check(square, 2u); check(seed, 4u);
        out[i] = seed;
        value = square / f32(p.len);
    }
    sums[lane] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) { sums[lane] = sums[lane] + sums[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u && group_index(wid) < p.partials) {
        check(sums[0], 8u);
        aux[group_index(wid)] = sums[0];
    }
}

@compute @workgroup_size(256)
fn mse_reduce(@builtin(local_invocation_index) lane: u32) {
    var value = 0.0;
    for (var i = lane; i < p.partials; i = i + 256u) {
        value = value + a[i]; check(value, 8u);
    }
    sums[lane] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (lane < stride) { sums[lane] = sums[lane] + sums[lane + stride]; }
        workgroupBarrier();
    }
    if (lane == 0u) { check(sums[0], 8u); aux[0] = sums[0]; }
}

@compute @workgroup_size(256)
fn prepare_sgd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i < p.len) {
        let change = step.rate * c[i];
        let candidate = a[i] - change;
        check(c[i], 4096u); check(change, 8192u); check(candidate, 16384u);
        out[i] = candidate;
    }
    if (i < p.cols) {
        let change = step.rate * d[i];
        let candidate = b[i] - change;
        check(d[i], 4096u); check(change, 8192u); check(candidate, 16384u);
        aux[i] = candidate;
    }
}

@compute @workgroup_size(1)
fn decide_sgd() {
    var flags = 0u;
    for (var i = 0u; i <= p.stages; i = i + 1u) {
        flags = flags | atomicLoad(&validation[i]);
    }
    atomicStore(&validation[p.stages + 1u], flags);
}

@compute @workgroup_size(256)
fn commit_sgd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    // Separate dispatch after the all-layer decision, never a workgroup-local vote.
    if (atomicLoad(&validation[p.stages + 1u]) != 0u || step.rate == 0.0) { return; }
    let i = index(wid, lane);
    if (i < p.len) { out[i] = a[i]; }
    if (i < p.cols) { aux[i] = b[i]; }
}
