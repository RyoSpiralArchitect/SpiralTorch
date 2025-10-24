// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

const WORKGROUP_SIZE: u32 = 256u;

struct Params {
    rows: u32,
    cols: u32,
    in_stride: u32,
    out_stride: u32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> shared_max: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main_cs(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) _global_id: vec3<u32>,
    @builtin(num_workgroups) _num_workgroups: vec3<u32>,
    @builtin(subgroup_size) subgroup_size: u32,
    @builtin(subgroup_invocation_id) subgroup_lane: u32,
) {
    let row = workgroup_id.x;
    if (row >= params.rows) {
        return;
    }

    let tid = local_id.x;
    let cols = params.cols;
    if (cols == 0u) {
        return;
    }

    let subgroup_count = (WORKGROUP_SIZE + subgroup_size - 1u) / subgroup_size;
    let subgroup_id = tid / subgroup_size;
    let row_in_base = row * params.in_stride;
    let row_out_base = row * params.out_stride;

    var local_max = -1e30;
    var idx = tid;
    loop {
        if (idx >= cols) {
            break;
        }
        let value = input[row_in_base + idx];
        local_max = max(local_max, value);
        idx += WORKGROUP_SIZE;
    }

    let sg_max = subgroupMax(local_max);

    if (subgroup_lane == 0u && subgroup_id < subgroup_count) {
        shared_max[subgroup_id] = sg_max;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var global_max = -1e30;
        for (var i = 0u; i < subgroup_count; i = i + 1u) {
            global_max = max(global_max, shared_max[i]);
        }
        shared_max[0] = global_max;
    }
    workgroupBarrier();
    let row_max = shared_max[0];

    var local_sum = 0.0;
    idx = tid;
    loop {
        if (idx >= cols) {
            break;
        }
        let offset = row_in_base + idx;
        let value = input[offset];
        let exp_value = exp(value - row_max);
        output[row_out_base + idx] = exp_value;
        local_sum = local_sum + exp_value;
        idx += WORKGROUP_SIZE;
    }

    let sg_sum = subgroupAdd(local_sum);
    if (subgroup_lane == 0u && subgroup_id < subgroup_count) {
        shared_sum[subgroup_id] = sg_sum;
    }
    workgroupBarrier();

    if (tid == 0u) {
        var global_sum = 0.0;
        for (var i = 0u; i < subgroup_count; i = i + 1u) {
            global_sum = global_sum + shared_sum[i];
        }
        shared_sum[0] = global_sum;
    }
    workgroupBarrier();

    let row_sum = shared_sum[0];
    let inv_sum = select(0.0, 1.0 / row_sum, row_sum > 0.0);

    idx = tid;
    loop {
        if (idx >= cols) {
            break;
        }
        let offset = row_out_base + idx;
        output[offset] = output[offset] * inv_sum;
        idx += WORKGROUP_SIZE;
    }
}
