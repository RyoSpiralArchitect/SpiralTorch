// Appended to dense_training.wgsl. Global norm of policy-normalized gradients.
// A pair is (largest magnitude, sum of squared values divided by that magnitude).
var<workgroup> norm_pairs: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn clip_partials(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    var value = vec2<f32>(0.0);
    if (i < p.len) {
        let g = effective(a[i]);
        let magnitude = bitcast<u32>(g) & 0x7fffffffu;
        if ((magnitude & 0x7f800000u) != 0x7f800000u && magnitude != 0u) {
            value = vec2<f32>(bitcast<f32>(magnitude), 1.0);
        }
    }
    norm_pairs[lane] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) { norm_pairs[lane] = norm_merge(norm_pairs[lane], norm_pairs[lane + stride]); }
        workgroupBarrier();
    }
    let group = group_index(wid);
    if (lane == 0u && group < p.partials) {
        let offset = 2u * (p.cols + group);
        out[offset] = norm_pairs[0].x;
        out[offset + 1u] = norm_pairs[0].y;
    }
}

@compute @workgroup_size(256)
fn clip_reduce(@builtin(local_invocation_index) lane: u32) {
    var value = vec2<f32>(0.0);
    for (var i = lane; i < p.partials; i += 256u) {
        value = norm_merge(value, vec2<f32>(a[2u * i], a[2u * i + 1u]));
    }
    norm_pairs[lane] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if (lane < stride) { norm_pairs[lane] = norm_merge(norm_pairs[lane], norm_pairs[lane + stride]); }
        workgroupBarrier();
    }
    if (lane == 0u) {
        var clipped = global_clip_factors(norm_pairs[0], step._pad1, i32(step._pad2));
        if (clipped.invalid != 0u) { atomicOr(&validation[p.stage], 4096u); }
        out[0] = f32(clipped.count);
        for (var i = 0u; i < clipped.count; i++) { out[1u + i] = clipped.values[i]; }
    }
}

@compute @workgroup_size(256)
fn clip_prepare(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = index(wid, lane);
    if (i < p.len) {
        let gradient = clipped_gradient(i);
        let candidate = sgd_candidate(a[i], gradient, step.rate, 4096u);
        if (candidate.flags != 0u) { atomicOr(&validation[p.stage], candidate.flags); }
        out[i] = candidate.value;
        aux[i] = gradient;
    }
}
