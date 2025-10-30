enable chromium_experimental_subgroups;

const WORKGROUP_SIZE: u32 = 256u;
const MAX_SUBGROUPS: u32 = WORKGROUP_SIZE;

struct Params {
    rows: u32,
    cols: u32,
    in_stride: u32,
    out_stride: u32,
    chimera_tile: u32,
    chimera_stripes: u32,
    flags: u32,
    mask_stride: u32,
};

const FLAG_CHIMERA: u32 = 1u;
const FLAG_HARDMAX_ONLY: u32 = 1u << 1u;
const FLAG_HARDMAX_MASK: u32 = 1u << 2u;

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@group(0) @binding(3)
var<storage, read_write> mask_output: array<f32>;

var<workgroup> shared_max: array<f32, MAX_SUBGROUPS>;
var<workgroup> shared_sum: array<f32, MAX_SUBGROUPS>;

fn row_offset(row: u32, stride: u32, idx: u32) -> u32 {
    if ((params.flags & FLAG_CHIMERA) != 0u) {
        let tile = max(params.chimera_tile, 1u);
        let stripes = max(params.chimera_stripes, 1u);
        let stripe = idx / tile;
        let within = idx % tile;
        return row * stride + within * stripes + stripe;
    }
    return row * stride + idx;
}

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

    let in_stride = params.in_stride;
    let out_stride = params.out_stride;

    var local_max = -1e30;
    var idx = tid;
    loop {
        if (idx >= cols) {
            break;
        }
        let value = input[row_offset(row, in_stride, idx)];
        local_max = max(local_max, value);
        idx += WORKGROUP_SIZE;
    }

    let sg_max = subgroupMax(local_max);
    let subgroup_id = tid / subgroup_size;

    if (subgroup_lane == 0u) {
        shared_max[subgroup_id] = sg_max;
    }
    workgroupBarrier();

    if (tid == 0u) {
        let subgroup_count = (WORKGROUP_SIZE + subgroup_size - 1u) / subgroup_size;
        var global_max = -1e30;
        for (var i = 0u; i < subgroup_count; i = i + 1u) {
            global_max = max(global_max, shared_max[i]);
        }
        shared_max[0] = global_max;
    }
    workgroupBarrier();

    let row_max = shared_max[0];

    let wants_mask = (params.flags & FLAG_HARDMAX_MASK) != 0u;
    let hardmax_only = (params.flags & FLAG_HARDMAX_ONLY) != 0u;

    if (hardmax_only) {
        idx = tid;
        loop {
            if (idx >= cols) {
                break;
            }
            let offset = row_offset(row, in_stride, idx);
            let value = input[offset];
            let is_peak = select(0.0, 1.0, value == row_max);
            let out_index = row_offset(row, out_stride, idx);
            output[out_index] = is_peak;
            if (wants_mask) {
                mask_output[row_offset(row, params.mask_stride, idx)] = is_peak;
            }
            idx += WORKGROUP_SIZE;
        }
        return;
    }

    var local_sum = 0.0;
    idx = tid;
    loop {
        if (idx >= cols) {
            break;
        }
        let offset = row_offset(row, in_stride, idx);
        let value = input[offset];
        let exp_value = exp(value - row_max);
        output[row_offset(row, out_stride, idx)] = exp_value;
        local_sum = local_sum + exp_value;
        idx += WORKGROUP_SIZE;
    }

    let sg_sum = subgroupAdd(local_sum);
    if (subgroup_lane == 0u) {
        shared_sum[subgroup_id] = sg_sum;
    }
    workgroupBarrier();

    if (tid == 0u) {
        let subgroup_count = (WORKGROUP_SIZE + subgroup_size - 1u) / subgroup_size;
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
        let offset = row_offset(row, out_stride, idx);
        output[offset] = output[offset] * inv_sum;
        idx += WORKGROUP_SIZE;
    }

    if (wants_mask) {
        idx = tid;
        loop {
            if (idx >= cols) {
                break;
            }
            let offset = row_offset(row, in_stride, idx);
            let value = input[offset];
            let is_peak = select(0.0, 1.0, value == row_max);
            mask_output[row_offset(row, params.mask_stride, idx)] = is_peak;
            idx += WORKGROUP_SIZE;
        }
    }
}
