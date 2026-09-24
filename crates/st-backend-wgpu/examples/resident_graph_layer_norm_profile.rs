//! GPU pass timing for one prepared affine LayerNorm training step.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use st_backend_wgpu::{
        resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
        resident_training::graph::ProfiledGraphTraining,
    };
    use st_kernel_contracts::{
        graph::{GraphDefinition, GraphGradientPolicy, GraphParameter, GraphStage, ParameterRole},
        layout::NdLayout,
    };

    let args: Vec<_> = std::env::args().skip(1).collect();
    if !args.is_empty()
        && (args.len() != 1 || (args[0] != "--paired" && args[0] != "--paired-reverse"))
    {
        return Err("usage: resident_graph_layer_norm_profile [--paired|--paired-reverse]".into());
    }
    let paired = !args.is_empty();
    let reverse = paired && args[0] == "--paired-reverse";
    let same_bits = |left: &[f32], right: &[f32]| {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    };

    let mut cases = Vec::new();
    for (rows, cols) in [(2, 3), (32, 256), (128, 1025)] {
        let definition = GraphDefinition::new(
            NdLayout::contiguous(&[rows, cols])?,
            vec![GraphStage::LayerNorm {
                gain: 0,
                bias: 1,
                epsilon: 1e-5,
            }],
            vec![
                GraphParameter {
                    role: ParameterRole::Gain,
                    shape: vec![cols],
                    values: vec![1.; cols],
                },
                GraphParameter {
                    role: ParameterRole::Bias,
                    shape: vec![cols],
                    values: vec![0.; cols],
                },
            ],
        )?;
        let mut graph = ProfiledGraphTraining::request_blocking(
            definition,
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
        for _ in 0..2 {
            graph.step(0.01)?;
            graph.loss_snapshot()?.read()?;
        }
        if paired {
            let (original, original_state, split, split_state) = if reverse {
                let split = graph.step_profiled_layer_norm_split(0.)?.read()?;
                let split_state = graph.state_snapshot()?.read()?;
                let original = graph.step_profiled(0.)?.read()?;
                let original_state = graph.state_snapshot()?.read()?;
                (original, original_state, split, split_state)
            } else {
                let original = graph.step_profiled(0.)?.read()?;
                let original_state = graph.state_snapshot()?.read()?;
                let split = graph.step_profiled_layer_norm_split(0.)?.read()?;
                let split_state = graph.state_snapshot()?.read()?;
                (original, original_state, split, split_state)
            };
            assert_eq!(original_state.loss.to_bits(), split_state.loss.to_bits());
            assert!(same_bits(
                &original_state.prediction,
                &split_state.prediction
            ));
            assert!(same_bits(
                &original_state.input_gradient,
                &split_state.input_gradient
            ));
            assert_eq!(
                original_state.raw_gradients.len(),
                split_state.raw_gradients.len()
            );
            assert_eq!(
                original_state.effective_gradients.len(),
                split_state.effective_gradients.len()
            );
            assert_eq!(
                original_state.graph.parameters().len(),
                split_state.graph.parameters().len()
            );
            for (left, right) in original_state
                .raw_gradients
                .iter()
                .zip(&split_state.raw_gradients)
            {
                assert!(same_bits(left, right));
            }
            for (left, right) in original_state
                .effective_gradients
                .iter()
                .zip(&split_state.effective_gradients)
            {
                assert!(same_bits(left, right));
            }
            for (left, right) in original_state
                .graph
                .parameters()
                .iter()
                .zip(split_state.graph.parameters())
            {
                assert!(same_bits(&left.values, &right.values));
            }
            cases.push(serde_json::json!({
                "rows": rows,
                "cols": cols,
                "adapter": format!("{:?}", graph.adapter_info()),
                "state_bits_equal": true,
                "order": if reverse { "split_first" } else { "original_first" },
                "original": original.report(),
                "split": split.report(),
            }));
        } else {
            let profile = graph.step_profiled(0.01)?.read()?;
            cases.push(serde_json::json!({
                "rows": rows,
                "cols": cols,
                "adapter": format!("{:?}", graph.adapter_info()),
                "report": profile.report(),
            }));
        }
    }
    println!("{}", serde_json::to_string_pretty(&cases)?);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("native GPU profile only");
}
