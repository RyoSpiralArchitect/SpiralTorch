const WORKGROUP_SIZE: u32 = 256u;

struct Params {
    rows: u32;
    cols: u32;
    in_stride: u32;
    out_stride: u32;
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> shared_max: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_sum: array<f32, WORKGROUP_SIZE>;

fn row_offset(row: u32, stride: u32, idx: u32) -> u32 {
    return row * stride + idx;
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

    shared_max[tid] = local_max;
    workgroupBarrier();
    reduce_max(tid, WORKGROUP_SIZE);

    let row_max = shared_max[0];

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
        let offset = row_offset(row, out_stride, idx);
        output[offset] = output[offset] * inv_sum;
        idx += WORKGROUP_SIZE;
    }
}
