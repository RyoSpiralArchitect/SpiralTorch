//! Matched synthetic GPU pass timings; readback is outside the timed passes.
//! Run the same probe binary/source on each revision. This is not an LLM benchmark.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use st_backend_wgpu::{
        resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
        resident_training::graph::ProfiledGraphTraining,
    };
    use st_kernel_contracts::{
        graph::{GraphDefinition, GraphGradientPolicy, GraphParameter, GraphStage, ParameterRole},
        layout::NdLayout,
        topos_resonator::ToposResonatorKernel,
    };

    let args: Vec<_> = std::env::args().skip(1).collect();
    if !args.is_empty() && args != ["--reverse"] {
        return Err("usage: topos_graph_profile [--reverse]".into());
    }
    let mut cases: Vec<_> = [(2, 3), (32, 256), (128, 1025)]
        .into_iter()
        .flat_map(|(rows, cols)| [1, 5, 64].map(|iterations| (rows, cols, iterations)))
        .collect();
    if !args.is_empty() {
        cases.reverse();
    }
    let bits = |values: &[f32]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut reports = Vec::new();
    for (rows, cols, iterations) in cases {
        let initial: Vec<_> = (0..cols).map(|i| (i % 11) as f32 * 0.17 - 0.8).collect();
        let graph = GraphDefinition::new(
            NdLayout::contiguous(&[rows, cols])?,
            vec![GraphStage::ToposResonator {
                gate: 0,
                kernel: ToposResonatorKernel::new(0.2, 1., 0.3, iterations)?,
                max_volume: rows * cols,
            }],
            vec![GraphParameter {
                role: ParameterRole::Gate,
                shape: vec![cols],
                values: initial.clone(),
            }],
        )?;
        let mut graph = ProfiledGraphTraining::request_blocking(
            graph,
            GraphGradientPolicy::Exact,
            MatmulTile::default(),
            MatmulKernel::Scalar,
            MatmulAccumulation::Sequential,
        )?;
        if graph.adapter_info().device_type == wgpu::DeviceType::Cpu {
            return Err("non-CPU GPU adapter required".into());
        }
        let input: Vec<_> = (0..rows * cols)
            .map(|i| (((i * 37 + 17) % 257) as f32 - 128.) / 64.)
            .collect();
        let target: Vec<_> = (0..rows * cols)
            .map(|i| (((i * 11 + 17) % 67) as f32 - 33.) / 64.)
            .collect();
        graph.upload_batch(&input, &target)?;
        for _ in 0..3 {
            graph.step(0.)?;
            graph.loss_snapshot()?.read()?;
        }
        let mut samples = Vec::new();
        for _ in 0..9 {
            samples.push(graph.step_profiled(0.)?.read()?.report());
        }
        let state = graph.state_snapshot()?.read()?;
        assert_eq!(bits(&state.graph.parameters()[0].values), bits(&initial));
        reports.push(serde_json::json!({
            "shape": [rows, cols], "iterations": iterations,
            "coupling": 0.2, "porosity": 0.3, "saturation": 1.0,
            "adapter": format!("{:?}", graph.adapter_info()),
            "warmups": 3, "learning_rate": 0.0, "samples": samples,
            "state_bits": {
                "loss": state.loss.to_bits(), "prediction": bits(&state.prediction),
                "input_gradient": bits(&state.input_gradient),
                "raw_gate_gradient": bits(&state.raw_gradients[0]),
                "effective_gate_gradient": bits(&state.effective_gradients[0]),
                "gate": bits(&state.graph.parameters()[0].values),
            },
        }));
    }
    println!("{}", serde_json::to_string(&reports)?);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("native GPU profile only");
}
