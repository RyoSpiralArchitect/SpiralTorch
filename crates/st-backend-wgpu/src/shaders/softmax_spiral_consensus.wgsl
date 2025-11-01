const WORKGROUP_SIZE: u32 = 256u;

struct Params {
    rows: u32,
    cols: u32,
    soft_stride: u32,
    mask_stride: u32,
    spiral_stride: u32,
    chimera_tile: u32,
    chimera_stripes: u32,
    flags: u32,
    phi: f32,
    phi_conjugate: f32,
    phi_bias: f32,
    leech_scale: f32,
    ramanujan_ratio: f32,
    inv_cols: f32,
    entropy_epsilon: f32,
    _pad: f32,
};

const FLAG_CHIMERA: u32 = 1u;

@group(0) @binding(0)
var<storage, read> softmax_input: array<f32>;

@group(0) @binding(1)
var<storage, read> hardmax_input: array<f32>;

@group(0) @binding(2)
var<storage, read_write> spiral_output: array<f32>;

@group(0) @binding(3)
var<storage, read_write> metrics_output: array<vec4<f32>>;

@group(0) @binding(4)
var<uniform> params: Params;

var<workgroup> shared_entropy: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_hardmass: array<f32, WORKGROUP_SIZE>;
var<workgroup> shared_scale: array<f32, WORKGROUP_SIZE>;

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

fn reduce_sum_entropy(local_id: u32) {
    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        workgroupBarrier();
        if (local_id < stride) {
            shared_entropy[local_id] =
                shared_entropy[local_id] + shared_entropy[local_id + stride];
        }
        stride = stride / 2u;
    }
    workgroupBarrier();
}

fn reduce_sum_hardmass(local_id: u32) {
    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        workgroupBarrier();
        if (local_id < stride) {
            shared_hardmass[local_id] =
                shared_hardmass[local_id] + shared_hardmass[local_id + stride];
        }
        stride = stride / 2u;
    }
    workgroupBarrier();
}

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = workgroup_id.x;
    if (row >= params.rows) {
        return;
    }

    let tid = local_id.x;
    let cols = params.cols;
    if (cols == 0u) {
        return;
    }

    let use_chimera = (params.flags & FLAG_CHIMERA) != 0u;
    let chimera_tile = select(1u, max(params.chimera_tile, 1u), use_chimera);
    let chimera_stripes = select(1u, max(params.chimera_stripes, 1u), use_chimera);
    let soft_base = row * params.soft_stride;
    let mask_base = row * params.mask_stride;
    let spiral_base = row * params.spiral_stride;

    var entropy = 0.0;
    var hardmass = 0.0;
    var idx = tid;
    loop {
        if (idx >= cols) {
            break;
        }
        let soft_index = layout_offset(soft_base, idx, use_chimera, chimera_tile, chimera_stripes);
        let mask_index = layout_offset(mask_base, idx, use_chimera, chimera_tile, chimera_stripes);
        let prob = max(softmax_input[soft_index], 0.0);
        let safe_prob = max(prob, params.entropy_epsilon);
        entropy = entropy - prob * log(safe_prob);
        hardmass = hardmass + max(hardmax_input[mask_index], 0.0);
        idx += WORKGROUP_SIZE;
    }

    shared_entropy[tid] = entropy;
    shared_hardmass[tid] = hardmass;
    workgroupBarrier();

    reduce_sum_entropy(tid);
    reduce_sum_hardmass(tid);

    let row_entropy = shared_entropy[0];
    let row_hardmass = shared_hardmass[0];

    if (tid == 0u) {
        let geodesic = row_entropy * params.ramanujan_ratio + row_hardmass * params.phi;
        var enrichment = 0.0;
        if (abs(geodesic) > params.entropy_epsilon) {
            enrichment = params.leech_scale * geodesic;
        }
        let scale = 1.0 + enrichment;
        let entropy_norm = clamp(row_entropy / (row_entropy + 1.0), 0.0, 1.0);
        let hardmass_norm = clamp(row_hardmass * params.inv_cols, 0.0, 1.0);
        let enrichment_norm = clamp(enrichment / (1.0 + abs(enrichment)), 0.0, 1.0);
        let coherence = (entropy_norm + hardmass_norm + enrichment_norm) / 3.0;

        shared_scale[0] = scale;
        shared_scale[1] = row_entropy;
        shared_scale[2] = row_hardmass;
        shared_scale[3] = enrichment;
        shared_scale[4] = coherence;
    }
    workgroupBarrier();

    let scale = shared_scale[0];
    let fused_entropy = shared_scale[1];
    let fused_hardmass = shared_scale[2];
    let fused_enrichment = shared_scale[3];
    let fused_coherence = shared_scale[4];

    idx = tid;
    loop {
        if (idx >= cols) {
            break;
        }
        let soft_index = layout_offset(soft_base, idx, use_chimera, chimera_tile, chimera_stripes);
        let mask_index = layout_offset(mask_base, idx, use_chimera, chimera_tile, chimera_stripes);
        let out_index = layout_offset(spiral_base, idx, use_chimera, chimera_tile, chimera_stripes);
        let prob = softmax_input[soft_index];
        let mask = hardmax_input[mask_index];
        let fused = params.phi_conjugate * prob + params.phi_bias * mask;
        spiral_output[out_index] = scale * fused;
        idx += WORKGROUP_SIZE;
    }

    if (tid == 0u) {
        metrics_output[row] = vec4<f32>(fused_entropy, fused_hardmass, fused_enrichment, fused_coherence);
    }
}
