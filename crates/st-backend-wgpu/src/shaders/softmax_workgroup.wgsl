const WORKGROUP_SIZE: u32 = 256u;

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

var<workgroup> shared_max: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

fn chimera_offset(idx: u32, tile: u32, stripes: u32) -> u32 {
    let stripe = idx / tile;
    let within = idx - stripe * tile;
    return within * stripes + stripe;
}

fn layout_offset(base: u32, idx: u32, use_chimera: bool, tile: u32, stripes: u32) -> u32 {
    if (use_chimera) {
        return base + chimera_offset(idx, tile, stripes);
    }
    return base + idx;
}

fn reduce_max(local_id: u32, active_threads: u32) {
    var stride = active_threads / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        workgroupBarrier();
        if (local_id < stride) {
            let other = shared_max[local_id + stride];
            shared_max[local_id] = max(shared_max[local_id], other);
        }
        stride = stride / 2u;
    }
    workgroupBarrier();
}

fn reduce_sum(local_id: u32, active_threads: u32) {
    var stride = active_threads / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        workgroupBarrier();
        if (local_id < stride) {
            let other = shared_sum[local_id + stride];
            shared_sum[local_id] = shared_sum[local_id] + other;
        }
        stride = stride / 2u;
    }
    workgroupBarrier();
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main_cs(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) _global_id: vec3<u32>,
    @builtin(num_workgroups) _num_workgroups: vec3<u32>,
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

    let use_chimera = (params.flags & FLAG_CHIMERA) != 0u;
    let chimera_tile = select(1u, max(params.chimera_tile, 1u), use_chimera);
    let chimera_stripes = select(1u, max(params.chimera_stripes, 1u), use_chimera);
    let row_in_base = row * in_stride;
    let row_out_base = row * out_stride;
    let wants_mask = (params.flags & FLAG_HARDMAX_MASK) != 0u;
    var row_mask_base = 0u;
    if (wants_mask) {
        row_mask_base = row * params.mask_stride;
    }

    var local_max = -1e30;
    var idx = tid;
    loop {
        if (idx >= cols) {
            break;
        }
        let value = input[layout_offset(row_in_base, idx, use_chimera, chimera_tile, chimera_stripes)];
        local_max = max(local_max, value);
        idx += WORKGROUP_SIZE;
    }

    shared_max[tid] = local_max;
    workgroupBarrier();
    reduce_max(tid, WORKGROUP_SIZE);

    let row_max = shared_max[0];

    let hardmax_only = (params.flags & FLAG_HARDMAX_ONLY) != 0u;

    if (hardmax_only) {
        idx = tid;
        loop {
            if (idx >= cols) {
                break;
            }
            let in_index = layout_offset(row_in_base, idx, use_chimera, chimera_tile, chimera_stripes);
            let value = input[in_index];
            let is_peak = select(0.0, 1.0, value == row_max);
            let out_index = layout_offset(row_out_base, idx, use_chimera, chimera_tile, chimera_stripes);
            output[out_index] = is_peak;
            if (wants_mask) {
                mask_output[layout_offset(row_mask_base, idx, use_chimera, chimera_tile, chimera_stripes)] = is_peak;
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
        let in_index = layout_offset(row_in_base, idx, use_chimera, chimera_tile, chimera_stripes);
        let value = input[in_index];
        let exp_value = exp(value - row_max);
        let out_index = layout_offset(row_out_base, idx, use_chimera, chimera_tile, chimera_stripes);
        output[out_index] = exp_value;
        if (wants_mask) {
            let is_peak = select(0.0, 1.0, value == row_max);
            mask_output[layout_offset(row_mask_base, idx, use_chimera, chimera_tile, chimera_stripes)] = is_peak;
        }
        local_sum = local_sum + exp_value;
        idx += WORKGROUP_SIZE;
    }

    shared_sum[tid] = local_sum;
    workgroupBarrier();
    reduce_sum(tid, WORKGROUP_SIZE);

    let row_sum = shared_sum[0];
    let inv_sum = select(0.0, 1.0 / row_sum, row_sum > 0.0);

    idx = tid;
    loop {
        if (idx >= cols) {
            break;
        }
        let offset = layout_offset(row_out_base, idx, use_chimera, chimera_tile, chimera_stripes);
        output[offset] = output[offset] * inv_sum;
        idx += WORKGROUP_SIZE;
    }
}
