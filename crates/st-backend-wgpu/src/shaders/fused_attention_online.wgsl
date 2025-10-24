const WORKGROUP_SIZE : u32 = {WORKGROUP_SIZE}u;
const MAX_HEAD_DIM : u32 = {MAX_HEAD_DIM}u;

struct Params {
    contexts: u32;
    sequence: u32;
    head_dim: u32;
    flags: u32;
    scale: f32;
    _pad0: f32;
    _pad1: f32;
    _pad2: f32;
};

const FLAG_USE_Z_BIAS : u32 = 1u << 0u;
const FLAG_USE_ATTN_BIAS : u32 = 1u << 1u;

@group(0) @binding(0) var<storage, read> queries : array<f32>;
@group(0) @binding(1) var<storage, read> keys : array<f32>;
@group(0) @binding(2) var<storage, read> values : array<f32>;
@group(0) @binding(3) var<storage, read> z_bias : array<f32>;
@group(0) @binding(4) var<storage, read> attn_bias : array<f32>;
@group(0) @binding(5) var<storage, read_write> output : array<f32>;
@group(0) @binding(6) var<uniform> params : Params;

var<workgroup> shared_q : array<f32, MAX_HEAD_DIM>;
var<workgroup> shared_k : array<f32, MAX_HEAD_DIM>;
var<workgroup> accum : array<f32, MAX_HEAD_DIM>;
var<workgroup> partials : array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_running_max : f32;
var<workgroup> shared_running_sum : f32;
var<workgroup> shared_alpha : f32;
var<workgroup> shared_weight : f32;
var<workgroup> shared_logit : f32;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn main(
    @builtin(workgroup_id) workgroup : vec3<u32>,
    @builtin(local_invocation_id) lid : vec3<u32>,
) {
    let context = workgroup.y;
    let query = workgroup.x;
    let lane = lid.x;
    let seq = params.sequence;
    let head_dim = params.head_dim;

    if (context >= params.contexts || query >= seq || head_dim == 0u) {
        return;
    }
    if (head_dim > MAX_HEAD_DIM) {
        return;
    }

    let context_offset = context * seq;
    let query_offset = (context_offset + query) * head_dim;

    var dim = lane;
    loop {
        if (dim >= head_dim) {
            break;
        }
        shared_q[dim] = queries[query_offset + dim];
        accum[dim] = 0.0;
        dim = dim + WORKGROUP_SIZE;
    }
    workgroupBarrier();

    if (lane == 0u) {
        shared_running_max = -3.40282347e38;
        shared_running_sum = 0.0;
    }
    workgroupBarrier();

    var key = 0u;
    loop {
        if (key >= seq) {
            break;
        }
        let key_offset = (context_offset + key) * head_dim;

        dim = lane;
        loop {
            if (dim >= head_dim) {
                break;
            }
            shared_k[dim] = keys[key_offset + dim];
            dim = dim + WORKGROUP_SIZE;
        }
        workgroupBarrier();

        var partial = 0.0;
        dim = lane;
        loop {
            if (dim >= head_dim) {
                break;
            }
            partial = partial + shared_q[dim] * shared_k[dim];
            dim = dim + WORKGROUP_SIZE;
        }
        partials[lane] = partial;
        workgroupBarrier();

        var active = WORKGROUP_SIZE;
        loop {
            if (active <= 1u) {
                break;
            }
            let half = (active + 1u) >> 1u;
            if (lane < half && lane + half < active) {
                partials[lane] = partials[lane] + partials[lane + half];
            }
            active = half;
            workgroupBarrier();
        }

        if (lane == 0u) {
            var logit = partials[0u] * params.scale;
            if ((params.flags & FLAG_USE_Z_BIAS) != 0u) {
                logit = logit + z_bias[context_offset + key];
            }
            if ((params.flags & FLAG_USE_ATTN_BIAS) != 0u) {
                let bias_index = (context_offset + query) * seq + key;
                logit = logit + attn_bias[bias_index];
            }
            shared_logit = logit;
        }
        workgroupBarrier();

        let logit = shared_logit;
        if (lane == 0u) {
            let old_max = shared_running_max;
            let old_sum = shared_running_sum;
            let new_max = max(old_max, logit);
            let scaled_sum = select(0.0, old_sum * exp(old_max - new_max), old_sum > 0.0);
            let exp_curr = exp(logit - new_max);
            let denom = scaled_sum + exp_curr;
            shared_running_max = new_max;
            shared_running_sum = denom;
            let alpha = select(0.0, scaled_sum / denom, denom > 0.0);
            let weight = select(0.0, exp_curr / denom, denom > 0.0);
            shared_alpha = alpha;
            shared_weight = weight;
        }
        workgroupBarrier();

        let alpha = shared_alpha;
        let weight = shared_weight;
        dim = lane;
        loop {
            if (dim >= head_dim) {
                break;
            }
            let value = values[key_offset + dim];
            accum[dim] = accum[dim] * alpha + weight * value;
            dim = dim + WORKGROUP_SIZE;
        }
        workgroupBarrier();

        key = key + 1u;
    }

    dim = lane;
    loop {
        if (dim >= head_dim) {
            break;
        }
        output[query_offset + dim] = accum[dim];
        dim = dim + WORKGROUP_SIZE;
    }
}
