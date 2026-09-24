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
        let profile = graph.step_profiled(0.01)?.read()?;
        cases.push(serde_json::json!({
            "rows": rows,
            "cols": cols,
            "adapter": format!("{:?}", graph.adapter_info()),
            "report": profile.report(),
        }));
    }
    println!("{}", serde_json::to_string_pretty(&cases)?);
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("native GPU profile only");
}
