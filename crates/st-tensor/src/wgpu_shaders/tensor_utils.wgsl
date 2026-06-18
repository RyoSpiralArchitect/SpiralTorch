struct TensorUtilParams {
    rows: u32,
    cols: u32,
    values: u32,
    flags: u32,
    scalar: f32,
    saturation: f32,
    porosity: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> aux: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: TensorUtilParams;

var<workgroup> scratch: array<f32, 256>;

fn finite_sign(value: f32) -> f32 {
    if (value < 0.0) {
        return -1.0;
    }
    return 1.0;
}

fn porous_mix(value: f32, saturation: f32, porosity: f32) -> f32 {
    if (value != value || saturation <= 0.0) {
        return 0.0;
    }
    let limit = abs(saturation);
    let magnitude = abs(value);
    if (magnitude <= limit) {
        return value;
    }
    let sign = finite_sign(value);
    if (porosity <= 0.00000011920929) {
        return sign * limit;
    }
    let bleed = (magnitude - limit) / (magnitude + limit);
    let absorb = min(porosity * 0.25, 1.0);
    let softened = limit * max(1.0 - absorb * min(bleed, 1.0), 0.0);
    return sign * softened;
}

fn porous_mix_backward_factor(value: f32, saturation: f32, porosity: f32) -> f32 {
    if (value != value || saturation <= 0.0) {
        return 0.0;
    }
    let limit = abs(saturation);
    let magnitude = abs(value);
    if (magnitude <= limit) {
        return 1.0;
    }
    if (porosity <= 0.00000011920929) {
        return 0.0;
    }
    let absorb = min(porosity * 0.25, 1.0);
    let denom = magnitude + limit;
    if (denom <= 0.00000011920929) {
        return 0.0;
    }
    return -2.0 * limit * limit * absorb / (denom * denom);
}

@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        output[idx] = input[idx] * params.scalar;
    }
}

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        output[idx] = input[idx] + aux[idx];
    }
}

@compute @workgroup_size(256)
fn hadamard(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        output[idx] = input[idx] * aux[idx];
    }
}

@compute @workgroup_size(256)
fn mul_row(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let col = idx % params.cols;
        output[idx] = input[idx] * aux[col];
    }
}

@compute @workgroup_size(256)
fn row_affine(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let col = idx % params.cols;
        output[idx] = input[idx] * aux[col] + aux[params.cols + col];
    }
}

@compute @workgroup_size(256)
fn dynamic_klein_gordon_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let col = idx % params.cols;
        let wave = input[idx];
        let mass = aux[col];
        let spin = aux[params.cols + col];
        let amplitude = tanh(wave);
        let kg_coeff = 1.0 - params.scalar * params.saturation - params.scalar * params.scalar * mass;
        let dirac_coeff = params.scalar * spin;
        output[idx] = wave * kg_coeff + dirac_coeff * amplitude;
    }
}

@compute @workgroup_size(256)
fn dynamic_klein_gordon_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let col = idx % params.cols;
        let wave = input[idx];
        let grad = aux[idx];
        let mass = aux[params.values + col];
        let spin = aux[params.values + params.cols + col];
        let amplitude = tanh(wave);
        let sech2 = 1.0 - amplitude * amplitude;
        let kg_coeff = 1.0 - params.scalar * params.saturation - params.scalar * params.scalar * mass;
        let dirac_coeff = params.scalar * spin;
        output[idx] = grad * (kg_coeff + dirac_coeff * sech2);
    }
    if (idx < params.cols) {
        var mass_sum = 0.0;
        var spin_sum = 0.0;
        var row = 0u;
        loop {
            if (row >= params.rows) {
                break;
            }
            let value_idx = row * params.cols + idx;
            let wave = input[value_idx];
            let grad = aux[value_idx];
            mass_sum = mass_sum + grad * (-params.scalar * params.scalar * wave);
            spin_sum = spin_sum + grad * (params.scalar * tanh(wave));
            row = row + 1u;
        }
        output[params.values + idx] = mass_sum;
        output[params.values + params.cols + idx] = spin_sum;
    }
}

@compute @workgroup_size(256)
fn dynamic_hamilton_jacobi_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let row = idx / params.cols;
        let col = idx % params.cols;
        let current = input[idx];
        var prev = current;
        if (row > 0u) {
            prev = input[idx - params.cols];
        }
        var next = current;
        if (row + 1u < params.rows) {
            next = input[idx + params.cols];
        }
        let potential = aux[col];
        let grad = (2.0 * current - prev - next) + potential * current;
        output[idx] = current - params.scalar * grad;
    }
}

@compute @workgroup_size(256)
fn dynamic_hamilton_jacobi_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let row = idx / params.cols;
        let col = idx % params.cols;
        let potential = aux[params.values + col];
        var factor = 1.0 - params.scalar * (2.0 + potential);
        if (params.rows == 1u) {
            factor = 1.0 - params.scalar * potential;
        } else if (row == 0u || row + 1u == params.rows) {
            factor = 1.0 - params.scalar * (1.0 + potential);
        }
        var grad = aux[idx] * factor;
        if (row > 0u) {
            grad = grad + aux[idx - params.cols] * params.scalar;
        }
        if (row + 1u < params.rows) {
            grad = grad + aux[idx + params.cols] * params.scalar;
        }
        output[idx] = grad;
    }
    if (idx < params.cols) {
        var potential_sum = 0.0;
        var row = 0u;
        loop {
            if (row >= params.rows) {
                break;
            }
            let value_idx = row * params.cols + idx;
            potential_sum = potential_sum + (-params.scalar * input[value_idx] * aux[value_idx]);
            row = row + 1u;
        }
        output[params.values + idx] = potential_sum;
    }
}

@compute @workgroup_size(256)
fn dynamic_schrodinger_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let col = idx % params.cols;
        let amp = tanh(input[idx]);
        let deco = 1.0 / (1.0 + params.scalar * abs(amp));
        output[idx] = amp * aux[col] * deco;
        output[params.values + idx] = amp;
        output[params.values * 2u + idx] = deco;
    }
}

@compute @workgroup_size(256)
fn dynamic_schrodinger_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let col = idx % params.cols;
        let amp = input[idx];
        let grad = aux[idx];
        let deco = aux[params.values + idx];
        let coherence = aux[params.values * 2u + col];
        let denom = 1.0 + params.scalar * abs(amp);
        let sign = finite_sign(amp);
        let d_deco_d_amp = -params.scalar * sign / (denom * denom);
        let base = deco + amp * d_deco_d_amp;
        let d_amp_d_input = 1.0 - amp * amp;
        output[idx] = grad * coherence * base * d_amp_d_input;
    }
    if (idx < params.cols) {
        var coherence_sum = 0.0;
        var row = 0u;
        loop {
            if (row >= params.rows) {
                break;
            }
            let value_idx = row * params.cols + idx;
            coherence_sum = coherence_sum + aux[value_idx] * input[value_idx] * aux[params.values + value_idx];
            row = row + 1u;
        }
        output[params.values + idx] = coherence_sum;
    }
}

@compute @workgroup_size(256)
fn sequence_last_step_gather(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let batch_idx = idx / params.cols;
        let col = idx % params.cols;
        let out_steps = params.flags;
        let src = batch_idx * out_steps * params.cols + (out_steps - 1u) * params.cols + col;
        output[idx] = input[src];
    }
}

@compute @workgroup_size(256)
fn sequence_last_step_scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let out_steps = params.flags;
        let col = idx % params.cols;
        let step = (idx / params.cols) % out_steps;
        let batch_idx = idx / (params.cols * out_steps);
        var value = 0.0;
        if (step + 1u == out_steps) {
            value = input[batch_idx * params.cols + col];
        }
        output[idx] = value;
    }
}

fn coherence_scan_steps() -> u32 {
    return params.flags >> 16u;
}

fn coherence_scan_memory() -> u32 {
    return params.flags & 65535u;
}

fn coherence_scan_order() -> f32 {
    return 1.0 + min(sqrt(-params.scalar), 4.0);
}

fn coherence_scan_score(batch_idx: u32, step: u32, steps: u32, dim: u32) -> f32 {
    let base = batch_idx * steps * dim;
    let query_offset = base + (steps - 1u) * dim;
    let value_offset = base + step * dim;
    var mse = 0.0;
    var col = 0u;
    loop {
        if (col >= dim) {
            break;
        }
        let diff = input[query_offset + col] - input[value_offset + col];
        mse = mse + diff * diff;
        col = col + 1u;
    }
    let mean = mse / max(f32(dim), 1.0);
    let dist = max(sqrt(mean) * sqrt(-params.scalar) / params.saturation, 0.000001);
    let denom = pow(dist, coherence_scan_order()) + 0.000000000001;
    let score = 1.0 / denom;
    if (score == score && abs(score) <= 3.402823e38) {
        return score;
    }
    return 0.0;
}

fn coherence_scan_total(batch_idx: u32, start_step: u32, steps: u32, dim: u32) -> f32 {
    var total = 0.0;
    var step = start_step;
    loop {
        if (step >= steps) {
            break;
        }
        var score = coherence_scan_score(batch_idx, step, steps, dim);
        if (step + 1u == steps) {
            score = score * params.porosity;
        }
        total = total + score;
        step = step + 1u;
    }
    return total;
}

fn coherence_scan_fallback_weight(step: u32, start_step: u32, steps: u32, memory: u32) -> f32 {
    if (step < start_step || step >= steps) {
        return 0.0;
    }
    let query_step = steps - 1u;
    if (params.porosity > 0.0) {
        return 1.0 / max(f32(memory), 1.0);
    }
    if (step == query_step && memory > 1u) {
        return 0.0;
    }
    return 1.0 / max(f32(max(memory - 1u, 1u)), 1.0);
}

fn coherence_scan_weight(batch_idx: u32, step: u32, start_step: u32, steps: u32, dim: u32, memory: u32, total: f32) -> f32 {
    if (step < start_step || step >= steps) {
        return 0.0;
    }
    if (!(total == total) || total <= 0.0 || abs(total) > 3.402823e38) {
        return coherence_scan_fallback_weight(step, start_step, steps, memory);
    }
    var score = coherence_scan_score(batch_idx, step, steps, dim);
    if (step + 1u == steps) {
        score = score * params.porosity;
    }
    return score / total;
}

fn coherence_scan_grad_output(batch_idx: u32, col: u32, dim: u32) -> f32 {
    return aux[batch_idx * dim + col];
}

fn coherence_scan_cached_weight(batch_idx: u32, step: u32, steps: u32, dim: u32) -> f32 {
    let weight_offset = params.rows * dim;
    return aux[weight_offset + batch_idx * steps + step];
}

fn coherence_scan_grad_value_dot(batch_idx: u32, step: u32, steps: u32, dim: u32) -> f32 {
    let base = batch_idx * steps * dim;
    var dot = 0.0;
    var col = 0u;
    loop {
        if (col >= dim) {
            break;
        }
        dot = dot + coherence_scan_grad_output(batch_idx, col, dim) * input[base + step * dim + col];
        col = col + 1u;
    }
    return dot;
}

fn coherence_scan_weighted_grad_dot(batch_idx: u32, start_step: u32, steps: u32, dim: u32) -> f32 {
    var weighted_dot = 0.0;
    var step = start_step;
    loop {
        if (step >= steps) {
            break;
        }
        let weight = coherence_scan_cached_weight(batch_idx, step, steps, dim);
        if (weight != 0.0) {
            weighted_dot = weighted_dot + weight * coherence_scan_grad_value_dot(batch_idx, step, steps, dim);
        }
        step = step + 1u;
    }
    return weighted_dot;
}

fn coherence_scan_score_gradient_common(batch_idx: u32, step: u32, steps: u32, dim: u32, score_scale: f32) -> f32 {
    if (score_scale == 0.0) {
        return 0.0;
    }
    let base = batch_idx * steps * dim;
    let query_offset = base + (steps - 1u) * dim;
    let value_offset = base + step * dim;
    var mse = 0.0;
    var col = 0u;
    loop {
        if (col >= dim) {
            break;
        }
        let diff = input[query_offset + col] - input[value_offset + col];
        mse = mse + diff * diff;
        col = col + 1u;
    }
    let dim_f = max(f32(dim), 1.0);
    let mean = mse / dim_f;
    if (!(mean == mean) || mean <= 0.0 || abs(mean) > 3.402823e38) {
        return 0.0;
    }
    let sqrt_mean = sqrt(mean);
    let alpha = sqrt(-params.scalar) / params.saturation;
    let dist = sqrt_mean * alpha;
    if (!(dist == dist) || dist <= 0.000001 || abs(dist) > 3.402823e38) {
        return 0.0;
    }
    let order = coherence_scan_order();
    let dist_pow = pow(dist, order);
    let denom = dist_pow + 0.000000000001;
    if (!(denom == denom) || denom <= 0.0 || abs(denom) > 3.402823e38) {
        return 0.0;
    }
    let dscore_ddist = -order * pow(dist, order - 1.0) / (denom * denom);
    let gradient_factor = score_scale * dscore_ddist * alpha / (dim_f * sqrt_mean);
    if (gradient_factor == gradient_factor && abs(gradient_factor) <= 3.402823e38) {
        return gradient_factor;
    }
    return 0.0;
}

@compute @workgroup_size(256)
fn zspace_coherence_scan_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.values) {
        return;
    }
    let batch = params.rows;
    let dim = params.cols;
    let steps = coherence_scan_steps();
    let memory = coherence_scan_memory();
    let context_values = batch * dim;
    let start_step = steps - memory;
    if (idx < context_values) {
        let batch_idx = idx / dim;
        let col = idx % dim;
        let total = coherence_scan_total(batch_idx, start_step, steps, dim);
        var value = 0.0;
        var step = start_step;
        loop {
            if (step >= steps) {
                break;
            }
            let weight = coherence_scan_weight(batch_idx, step, start_step, steps, dim, memory, total);
            let input_idx = batch_idx * steps * dim + step * dim + col;
            value = value + weight * input[input_idx];
            step = step + 1u;
        }
        if (params._pad2 > 0.0) {
            let query_idx = batch_idx * steps * dim + (steps - 1u) * dim + col;
            value = value + params._pad2 * input[query_idx];
        }
        output[idx] = value;
    } else {
        let weight_idx = idx - context_values;
        let batch_idx = weight_idx / steps;
        let step = weight_idx % steps;
        let total = coherence_scan_total(batch_idx, start_step, steps, dim);
        output[idx] = coherence_scan_weight(batch_idx, step, start_step, steps, dim, memory, total);
    }
}

@compute @workgroup_size(256)
fn zspace_coherence_scan_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.values) {
        return;
    }
    let dim = params.cols;
    let steps = coherence_scan_steps();
    let memory = coherence_scan_memory();
    let start_step = steps - memory;
    let query_step = steps - 1u;
    let col = idx % dim;
    let step = (idx / dim) % steps;
    let batch_idx = idx / (steps * dim);
    let base = batch_idx * steps * dim;
    let grad_output = coherence_scan_grad_output(batch_idx, col, dim);

    var grad = coherence_scan_cached_weight(batch_idx, step, steps, dim) * grad_output;
    if (step == query_step && params._pad2 > 0.0) {
        grad = grad + params._pad2 * grad_output;
    }

    let total = coherence_scan_total(batch_idx, start_step, steps, dim);
    if (total == total && total > 0.0 && abs(total) <= 3.402823e38) {
        let weighted_dot = coherence_scan_weighted_grad_dot(batch_idx, start_step, steps, dim);
        var score_step = start_step;
        loop {
            if (score_step >= steps) {
                break;
            }
            var score_scale = 1.0;
            var score = coherence_scan_score(batch_idx, score_step, steps, dim);
            if (score_step == query_step) {
                score_scale = params.porosity;
                score = score * score_scale;
            }
            if (score != 0.0) {
                let dot = coherence_scan_grad_value_dot(batch_idx, score_step, steps, dim);
                let dloss_dscore = (dot - weighted_dot) / total;
                if (dloss_dscore == dloss_dscore && dloss_dscore != 0.0 && abs(dloss_dscore) <= 3.402823e38) {
                    let gradient_factor = coherence_scan_score_gradient_common(batch_idx, score_step, steps, dim, score_scale);
                    if (gradient_factor != 0.0) {
                        let query_value = input[base + query_step * dim + col];
                        let source_value = input[base + score_step * dim + col];
                        let contribution = dloss_dscore * gradient_factor * (query_value - source_value);
                        if (contribution == contribution && abs(contribution) <= 3.402823e38) {
                            if (step == query_step) {
                                grad = grad + contribution;
                            }
                            if (step == score_step) {
                                grad = grad - contribution;
                            }
                        }
                    }
                }
            }
            score_step = score_step + 1u;
        }
    }
    output[idx] = grad;
}

@compute @workgroup_size(256)
fn add_scaled(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        output[idx] = input[idx] + aux[idx] * params.scalar;
    }
}

@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        output[idx] = input[idx] - aux[idx];
    }
}

@compute @workgroup_size(256)
fn transpose(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let row = idx / params.cols;
        let col = idx % params.cols;
        output[col * params.rows + row] = input[idx];
    }
}

@compute @workgroup_size(256)
fn embedding_gather(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let token_offset = idx / params.cols;
        let dim = idx % params.cols;
        let token_idx = u32(input[token_offset]);
        output[idx] = aux[token_idx * params.cols + dim];
    }
}

@compute @workgroup_size(256)
fn embedding_scatter_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let vocab_idx = idx / params.cols;
        let dim = idx % params.cols;
        var sum = 0.0;
        var token_offset = 0u;
        loop {
            if (token_offset >= params.rows) {
                break;
            }
            if (u32(input[token_offset]) == vocab_idx) {
                sum = sum + aux[token_offset * params.cols + dim];
            }
            token_offset = token_offset + 1u;
        }
        output[idx] = sum;
    }
}

@compute @workgroup_size(256)
fn hypergrad_accumulate_wave(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let value = aux[idx];
        let denom = 1.0 - params.scalar * value * value;
        let update = value / max(abs(denom), params._pad2);
        output[idx] = porous_mix(input[idx] + update, params.saturation, params.porosity);
    }
}

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        output[idx] = max(input[idx], 0.0);
    }
}

@compute @workgroup_size(256)
fn relu_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        output[idx] = select(0.0, aux[idx], input[idx] > 0.0);
    }
}

fn pool_channels() -> u32 { return u32(aux[0]); }
fn pool_input_h() -> u32 { return u32(aux[1]); }
fn pool_input_w() -> u32 { return u32(aux[2]); }
fn pool_output_h() -> u32 { return u32(aux[3]); }
fn pool_output_w() -> u32 { return u32(aux[4]); }
fn pool_kernel_h() -> u32 { return u32(aux[5]); }
fn pool_kernel_w() -> u32 { return u32(aux[6]); }
fn pool_stride_h() -> u32 { return u32(aux[7]); }
fn pool_stride_w() -> u32 { return u32(aux[8]); }
fn pool_pad_h() -> u32 { return u32(aux[9]); }
fn pool_pad_w() -> u32 { return u32(aux[10]); }
fn pool_config_len() -> u32 { return 11u; }

fn max_pool_best_index(
    row_offset: u32,
    channel_offset: u32,
    out_h_index: u32,
    out_w_index: u32,
) -> u32 {
    let input_h = pool_input_h();
    let input_w = pool_input_w();
    let kernel_h = pool_kernel_h();
    let kernel_w = pool_kernel_w();
    let stride_h = pool_stride_h();
    let stride_w = pool_stride_w();
    let pad_h = pool_pad_h();
    let pad_w = pool_pad_w();
    var best = -3.402823e38;
    var best_idx = channel_offset;
    var kh = 0u;
    loop {
        if (kh >= kernel_h) {
            break;
        }
        let pos_h = out_h_index * stride_h + kh;
        if (pos_h >= pad_h) {
            let input_h_index = pos_h - pad_h;
            if (input_h_index < input_h) {
                var kw = 0u;
                loop {
                    if (kw >= kernel_w) {
                        break;
                    }
                    let pos_w = out_w_index * stride_w + kw;
                    if (pos_w >= pad_w) {
                        let input_w_index = pos_w - pad_w;
                        if (input_w_index < input_w) {
                            let candidate_idx = channel_offset + input_h_index * input_w + input_w_index;
                            let value = input[row_offset + candidate_idx];
                            if (value > best) {
                                best = value;
                                best_idx = candidate_idx;
                            }
                        }
                    }
                    kw = kw + 1u;
                }
            }
        }
        kh = kh + 1u;
    }
    return best_idx;
}

@compute @workgroup_size(256)
fn max_pool2d_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let channels = pool_channels();
        let input_h = pool_input_h();
        let input_w = pool_input_w();
        let output_h = pool_output_h();
        let output_w = pool_output_w();
        let input_cols = channels * input_h * input_w;
        let output_cols = channels * output_h * output_w;
        let batch_index = idx / output_cols;
        let within_output = idx % output_cols;
        let channel_index = within_output / (output_h * output_w);
        let spatial = within_output % (output_h * output_w);
        let out_h_index = spatial / output_w;
        let out_w_index = spatial % output_w;
        let row_offset = batch_index * input_cols;
        let channel_offset = channel_index * input_h * input_w;
        let best_idx = max_pool_best_index(row_offset, channel_offset, out_h_index, out_w_index);
        output[idx] = input[row_offset + best_idx];
        output[params.values + idx] = f32(best_idx);
    }
}

@compute @workgroup_size(256)
fn max_pool2d_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let channels = pool_channels();
        let input_h = pool_input_h();
        let input_w = pool_input_w();
        let output_h = pool_output_h();
        let output_w = pool_output_w();
        let input_cols = channels * input_h * input_w;
        let output_cols = channels * output_h * output_w;
        let batch_index = idx / input_cols;
        let within_input = idx % input_cols;
        let channel_index = within_input / (input_h * input_w);
        let input_spatial = within_input % (input_h * input_w);
        let input_h_index = input_spatial / input_w;
        let input_w_index = input_spatial % input_w;
        let row_offset = batch_index * input_cols;
        let channel_offset = channel_index * input_h * input_w;
        let grad_offset = pool_config_len();
        var acc = 0.0;
        var out_h_index = 0u;
        loop {
            if (out_h_index >= output_h) {
                break;
            }
            var out_w_index = 0u;
            loop {
                if (out_w_index >= output_w) {
                    break;
                }
                let window_h_start = out_h_index * pool_stride_h();
                let window_w_start = out_w_index * pool_stride_w();
                let pos_h = input_h_index + pool_pad_h();
                let pos_w = input_w_index + pool_pad_w();
                if (
                    pos_h >= window_h_start &&
                    pos_w >= window_w_start &&
                    pos_h < window_h_start + pool_kernel_h() &&
                    pos_w < window_w_start + pool_kernel_w()
                ) {
                    let best_idx = max_pool_best_index(row_offset, channel_offset, out_h_index, out_w_index);
                    if (best_idx == within_input) {
                        let grad_idx =
                            batch_index * output_cols +
                            channel_index * output_h * output_w +
                            out_h_index * output_w +
                            out_w_index;
                        acc = acc + aux[grad_offset + grad_idx];
                    }
                }
                out_w_index = out_w_index + 1u;
            }
            out_h_index = out_h_index + 1u;
        }
        output[idx] = acc;
    }
}

@compute @workgroup_size(256)
fn avg_pool2d_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let channels = pool_channels();
        let input_h = pool_input_h();
        let input_w = pool_input_w();
        let output_h = pool_output_h();
        let output_w = pool_output_w();
        let input_cols = channels * input_h * input_w;
        let output_cols = channels * output_h * output_w;
        let batch_index = idx / output_cols;
        let within_output = idx % output_cols;
        let channel_index = within_output / (output_h * output_w);
        let spatial = within_output % (output_h * output_w);
        let out_h_index = spatial / output_w;
        let out_w_index = spatial % output_w;
        let row_offset = batch_index * input_cols;
        let channel_offset = channel_index * input_h * input_w;
        var acc = 0.0;
        var kh = 0u;
        loop {
            if (kh >= pool_kernel_h()) {
                break;
            }
            let pos_h = out_h_index * pool_stride_h() + kh;
            if (pos_h >= pool_pad_h()) {
                let input_h_index = pos_h - pool_pad_h();
                if (input_h_index < input_h) {
                    var kw = 0u;
                    loop {
                        if (kw >= pool_kernel_w()) {
                            break;
                        }
                        let pos_w = out_w_index * pool_stride_w() + kw;
                        if (pos_w >= pool_pad_w()) {
                            let input_w_index = pos_w - pool_pad_w();
                            if (input_w_index < input_w) {
                                let candidate_idx = channel_offset + input_h_index * input_w + input_w_index;
                                acc = acc + input[row_offset + candidate_idx];
                            }
                        }
                        kw = kw + 1u;
                    }
                }
            }
            kh = kh + 1u;
        }
        output[idx] = acc / f32(pool_kernel_h() * pool_kernel_w());
    }
}

@compute @workgroup_size(256)
fn avg_pool2d_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let channels = pool_channels();
        let input_h = pool_input_h();
        let input_w = pool_input_w();
        let output_h = pool_output_h();
        let output_w = pool_output_w();
        let input_cols = channels * input_h * input_w;
        let output_cols = channels * output_h * output_w;
        let batch_index = idx / input_cols;
        let within_input = idx % input_cols;
        let channel_index = within_input / (input_h * input_w);
        let input_spatial = within_input % (input_h * input_w);
        let input_h_index = input_spatial / input_w;
        let input_w_index = input_spatial % input_w;
        let grad_offset = pool_config_len();
        let area = f32(pool_kernel_h() * pool_kernel_w());
        var acc = 0.0;
        var out_h_index = 0u;
        loop {
            if (out_h_index >= output_h) {
                break;
            }
            var out_w_index = 0u;
            loop {
                if (out_w_index >= output_w) {
                    break;
                }
                let window_h_start = out_h_index * pool_stride_h();
                let window_w_start = out_w_index * pool_stride_w();
                let pos_h = input_h_index + pool_pad_h();
                let pos_w = input_w_index + pool_pad_w();
                if (
                    pos_h >= window_h_start &&
                    pos_w >= window_w_start &&
                    pos_h < window_h_start + pool_kernel_h() &&
                    pos_w < window_w_start + pool_kernel_w()
                ) {
                    let grad_idx =
                        batch_index * output_cols +
                        channel_index * output_h * output_w +
                        out_h_index * output_w +
                        out_w_index;
                    acc = acc + aux[grad_offset + grad_idx] / area;
                }
                out_w_index = out_w_index + 1u;
            }
            out_h_index = out_h_index + 1u;
        }
        output[idx] = acc;
    }
}

@compute @workgroup_size(256)
fn sum_squares(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var sum = 0.0;
    var idx = lane;
    loop {
        if (idx >= params.values) {
            break;
        }
        let value = input[idx];
        sum = sum + value * value;
        idx = idx + 256u;
    }
    scratch[lane] = sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[0] = scratch[0];
    }
}

@compute @workgroup_size(256)
fn sum_abs(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var sum = 0.0;
    var idx = lane;
    loop {
        if (idx >= params.values) {
            break;
        }
        sum = sum + abs(input[idx]);
        idx = idx + 256u;
    }
    scratch[lane] = sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[0] = scratch[0];
    }
}

fn probability_clamp(value: f32, epsilon: f32) -> f32 {
    return min(max(value, epsilon), 1.0);
}

fn binary_probability_clamp(value: f32, epsilon: f32) -> f32 {
    return min(max(value, epsilon), 1.0 - epsilon);
}

fn stable_softplus(value: f32) -> f32 {
    if (value > 0.0) {
        return value + log(1.0 + exp(-value));
    }
    return log(1.0 + exp(value));
}

fn stable_sigmoid(value: f32) -> f32 {
    if (value >= 0.0) {
        return 1.0 / (1.0 + exp(-value));
    }
    let exp_value = exp(value);
    return exp_value / (1.0 + exp_value);
}

@compute @workgroup_size(256)
fn lstm_forward_gate_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let unit = global_id.x;
    let hidden_dim = params.cols / 4u;
    if (unit < hidden_dim) {
        let i = stable_sigmoid(input[unit]);
        let f = stable_sigmoid(input[hidden_dim + unit]);
        let g = tanh(input[2u * hidden_dim + unit]);
        let o = stable_sigmoid(input[3u * hidden_dim + unit]);
        let cell = f * aux[unit] + i * g;
        let hidden = o * tanh(cell);
        output[unit] = i;
        output[hidden_dim + unit] = f;
        output[2u * hidden_dim + unit] = g;
        output[3u * hidden_dim + unit] = o;
        output[4u * hidden_dim + unit] = cell;
        output[5u * hidden_dim + unit] = hidden;
    }
}

@compute @workgroup_size(256)
fn categorical_cross_entropy_forward(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var sum = 0.0;
    var idx = lane;
    loop {
        if (idx >= params.values) {
            break;
        }
        let probability = probability_clamp(input[idx], params.scalar);
        sum = sum - aux[idx] * log(probability);
        idx = idx + 256u;
    }
    scratch[lane] = sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[0] = scratch[0] * params.saturation;
    }
}

@compute @workgroup_size(256)
fn categorical_cross_entropy_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let probability = probability_clamp(input[idx], params.scalar);
        output[idx] = (-aux[idx] / probability) * params.saturation;
    }
}

@compute @workgroup_size(256)
fn hyperbolic_cross_entropy_forward(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var sum = 0.0;
    var idx = lane;
    loop {
        if (idx >= params.values) {
            break;
        }
        let label_value = binary_probability_clamp(aux[idx], params.saturation);
        let scaled = input[idx] * params.scalar;
        let log_sigmoid_pos = -stable_softplus(-scaled);
        let log_sigmoid_neg = -stable_softplus(scaled);
        sum = sum + (-label_value * log_sigmoid_pos - (1.0 - label_value) * log_sigmoid_neg);
        idx = idx + 256u;
    }
    scratch[lane] = sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[0] = scratch[0] * params.porosity;
    }
}

@compute @workgroup_size(256)
fn hyperbolic_cross_entropy_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let label_value = binary_probability_clamp(aux[idx], params.saturation);
        let scaled = input[idx] * params.scalar;
        output[idx] = params.scalar * (stable_sigmoid(scaled) - label_value) * params.porosity;
    }
}

@compute @workgroup_size(256)
fn mse_loss_forward(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let lane = local_id.x;
    var sum = 0.0;
    var idx = lane;
    loop {
        if (idx >= params.values) {
            break;
        }
        let diff = input[idx] - aux[idx];
        sum = sum + diff * diff;
        idx = idx + 256u;
    }
    scratch[lane] = sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[0] = scratch[0] * params.scalar;
    }
}

@compute @workgroup_size(256)
fn mse_loss_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        output[idx] = (input[idx] - aux[idx]) * params.scalar;
    }
}

@compute @workgroup_size(256)
fn zspace_softmax_backward_fixed(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row = workgroup_id.x;
    let lane = local_id.x;
    let row_offset = row * params.cols;

    var row_max = -3.402823e38;
    var col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        row_max = max(row_max, input[row_offset + col] * params.scalar);
        col = col + 256u;
    }
    scratch[lane] = row_max;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = max(scratch[lane], scratch[lane + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let max_logit = scratch[0];

    var exp_sum = 0.0;
    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        exp_sum = exp_sum + exp(input[row_offset + col] * params.scalar - max_logit);
        col = col + 256u;
    }
    scratch[lane] = exp_sum;
    workgroupBarrier();

    stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let denom = max(scratch[0], 0.000000000001);

    var dot = 0.0;
    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let idx = row_offset + col;
        let probability = exp(input[idx] * params.scalar - max_logit) / denom;
        dot = dot + aux[idx] * probability;
        col = col + 256u;
    }
    scratch[lane] = dot;
    workgroupBarrier();

    stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_dot = scratch[0];

    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let idx = row_offset + col;
        let probability = exp(input[idx] * params.scalar - max_logit) / denom;
        output[idx] = params.scalar * probability * (aux[idx] - row_dot);
        col = col + 256u;
    }
}

@compute @workgroup_size(256)
fn add_row(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < params.values) {
        let col = idx % params.cols;
        output[idx] = input[idx] + aux[col];
    }
}

@compute @workgroup_size(256)
fn sum_axis0(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let col = workgroup_id.x;
    let lane = local_id.x;
    var sum = 0.0;
    var row = lane;
    loop {
        if (row >= params.rows) {
            break;
        }
        sum = sum + input[row * params.cols + col];
        row = row + 256u;
    }
    scratch[lane] = sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[col] = scratch[0];
    }
}

@compute @workgroup_size(256)
fn sum_axis0_scaled(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let col = workgroup_id.x;
    let lane = local_id.x;
    var sum = 0.0;
    var row = lane;
    loop {
        if (row >= params.rows) {
            break;
        }
        sum = sum + input[row * params.cols + col];
        row = row + 256u;
    }
    scratch[lane] = sum;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        output[col] = scratch[0] * params.scalar;
    }
}

@compute @workgroup_size(256)
fn project_to_poincare(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row_index = workgroup_id.x;
    let lane = local_id.x;
    var sum_sq = 0.0;
    var col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let value = input[row_index * params.cols + col];
        sum_sq = sum_sq + value * value;
        col = col + 256u;
    }
    scratch[lane] = sum_sq;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let norm = sqrt(scratch[0]);
    var factor = 1.0;
    if (norm > 0.0) {
        factor = tanh(norm / params.scalar) / norm;
    }

    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let idx = row_index * params.cols + col;
        output[idx] = input[idx] * factor;
        col = col + 256u;
    }
}

@compute @workgroup_size(256)
fn wave_gate_project(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row_index = workgroup_id.x;
    let lane = local_id.x;
    var sum_sq = 0.0;
    var col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let raw = input[row_index * params.cols + col] * aux[col] + aux[params.cols + col];
        let value = porous_mix(raw, params.saturation, params.porosity);
        sum_sq = sum_sq + value * value;
        col = col + 256u;
    }
    scratch[lane] = sum_sq;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let norm = sqrt(scratch[0]);
    var factor = 1.0;
    if (norm > 0.0) {
        factor = tanh(norm / params.scalar) / norm;
    }

    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let idx = row_index * params.cols + col;
        let raw = input[idx] * aux[col] + aux[params.cols + col];
        let value = porous_mix(raw, params.saturation, params.porosity);
        output[idx] = value * factor;
        col = col + 256u;
    }
}

@compute @workgroup_size(256)
fn wave_gate_backward(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let row_index = workgroup_id.x;
    let lane = local_id.x;
    var sum_sq = 0.0;
    var dot = 0.0;
    var col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let idx = row_index * params.cols + col;
        let raw = input[idx] * aux[params.values + col] + aux[params.values + params.cols + col];
        let preprojected = porous_mix(raw, params.saturation, params.porosity);
        sum_sq = sum_sq + preprojected * preprojected;
        dot = dot + preprojected * aux[idx];
        col = col + 256u;
    }
    scratch[lane] = sum_sq;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let norm = sqrt(scratch[0]);

    scratch[lane] = dot;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (lane < stride) {
            scratch[lane] = scratch[lane] + scratch[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let row_dot = scratch[0];

    col = lane;
    loop {
        if (col >= params.cols) {
            break;
        }
        let idx = row_index * params.cols + col;
        let gate = aux[params.values + col];
        let raw = input[idx] * gate + aux[params.values + params.cols + col];
        let preprojected = porous_mix(raw, params.saturation, params.porosity);
        var grad_saturated = aux[idx];
        if (norm <= params._pad2 || norm != norm) {
            grad_saturated = aux[idx] / params.scalar;
        } else {
            let t = tanh(norm / params.scalar);
            let factor = t / norm;
            let sech2 = 1.0 - t * t;
            let radial = ((sech2 * norm / params.scalar) - t) / (norm * norm * norm);
            grad_saturated = factor * aux[idx] + radial * preprojected * row_dot;
        }
        let grad_affine =
            grad_saturated * porous_mix_backward_factor(raw, params.saturation, params.porosity);
        output[idx] = grad_affine * gate;
        output[params.values + idx] = grad_affine;
        col = col + 256u;
    }
}
