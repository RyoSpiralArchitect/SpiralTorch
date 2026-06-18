struct LstmBackwardScanParams {
    timesteps: u32,
    hidden_dim: u32,
    gate_width: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> gates_i: array<f32>;
@group(0) @binding(1) var<storage, read> gates_f: array<f32>;
@group(0) @binding(2) var<storage, read> gates_g: array<f32>;
@group(0) @binding(3) var<storage, read> gates_o: array<f32>;
@group(0) @binding(4) var<storage, read> cell_states: array<f32>;
@group(0) @binding(5) var<storage, read> grad_output: array<f32>;
@group(0) @binding(6) var<storage, read> weight_hh_t: array<f32>;
@group(0) @binding(7) var<storage, read_write> gate_gradients: array<f32>;
@group(0) @binding(8) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(9) var<uniform> params: LstmBackwardScanParams;

@compute @workgroup_size(64)
fn lstm_backward_scan(@builtin(local_invocation_id) local_id: vec3<u32>) {
    if (params.timesteps == 0u || params.hidden_dim == 0u) {
        return;
    }

    let lane = local_id.x;
    var step = params.timesteps;
    loop {
        if (step == 0u) {
            break;
        }
        step = step - 1u;

        let hidden_base = step * params.hidden_dim;
        let prev_cell_base = step * params.hidden_dim;
        let curr_cell_base = (step + 1u) * params.hidden_dim;
        let gate_base = step * params.gate_width;

        var unit = lane;
        loop {
            if (unit >= params.hidden_dim) {
                break;
            }
            let value_idx = hidden_base + unit;
            let dh = grad_output[value_idx] + scratch[unit];
            let o = gates_o[value_idx];
            let i = gates_i[value_idx];
            let f = gates_f[value_idx];
            let g = gates_g[value_idx];
            let prev_cell = cell_states[prev_cell_base + unit];
            let curr_cell = cell_states[curr_cell_base + unit];
            let tanh_c = tanh(curr_cell);
            let do_gate = dh * tanh_c * o * (1.0 - o);
            let dc_recurrent = dh * o * (1.0 - tanh_c * tanh_c);
            let dc = dc_recurrent + scratch[params.hidden_dim + unit];
            let di = dc * g * i * (1.0 - i);
            let dg = dc * i * (1.0 - g * g);
            let df = dc * prev_cell * f * (1.0 - f);

            scratch[params.hidden_dim + unit] = dc * f;
            gate_gradients[gate_base + unit] = di;
            gate_gradients[gate_base + params.hidden_dim + unit] = df;
            gate_gradients[gate_base + 2u * params.hidden_dim + unit] = dg;
            gate_gradients[gate_base + 3u * params.hidden_dim + unit] = do_gate;
            unit = unit + 64u;
        }

        storageBarrier();

        unit = lane;
        loop {
            if (unit >= params.hidden_dim) {
                break;
            }
            var sum = 0.0;
            var gate = 0u;
            loop {
                if (gate >= params.gate_width) {
                    break;
                }
                sum = sum + gate_gradients[gate_base + gate]
                    * weight_hh_t[gate * params.hidden_dim + unit];
                gate = gate + 1u;
            }
            scratch[unit] = sum;
            unit = unit + 64u;
        }

        storageBarrier();
    }
}
