//! Native/browser admission checks outside every timed interval.
use super::*;
use st_backend_wgpu::resident_tensor::TensorDevice;

pub async fn run(runtime: WgpuRuntime) -> Result<usize> {
    let device = TensorDevice::new(runtime.clone())?;
    let values: Vec<_> = (0..72).map(|i| (i as f32 - 30.) / 32.).collect();
    let base = device.upload(&[2, 3, 12], &values)?;
    let views = [
        base.narrow(2, 0, 3)?,
        base.narrow(2, 4, 3)?,
        base.reshape(&[3, 2, 12])?
            .permute(&[1, 0, 2])?
            .narrow(2, 1, 3)?,
        base.reshape(&[3, 2, 12])?
            .permute(&[1, 2, 0])?
            .narrow(1, 0, 3)?,
        base.narrow(2, 0, 1)?.broadcast_to(&[2, 3, 3])?,
        base.narrow(0, 0, 1)?
            .narrow(1, 0, 1)?
            .narrow(2, 3, 3)?
            .broadcast_to(&[2, 3, 3])?,
    ];
    let definition = GraphDefinition::new(
        NdLayout::contiguous(&[2, 3, 3])?,
        vec![GraphStage::Pointwise {
            chain: PointwiseChain::new(
                3,
                vec![
                    PointwiseStep::named("multiply", Some(1))?,
                    PointwiseStep::named("add", Some(2))?,
                    PointwiseStep::named("add", Some(0))?,
                    PointwiseStep::named("relu", None)?,
                ],
            )?,
            parameters: vec![0, 1],
        }],
        vec![
            GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![3],
                values: vec![0.25, -0.5, 1.],
            },
            GraphParameter {
                role: ParameterRole::Bias,
                shape: vec![1],
                values: vec![-0.125],
            },
        ],
    )?;
    let mut graph = ResidentGraph::new(
        runtime,
        definition,
        MatmulTile::default(),
        MatmulKernel::Register2x2,
        MatmulAccumulation::Sequential,
    )?;
    let mut held = Vec::new();
    for view in &views {
        let expected: Vec<_> = (0..18)
            .map(|i| {
                let x = values[view.layout().storage_index(i).unwrap()];
                ((x * [0.25, -0.5, 1.][i % 3] - 0.125) + x).max(0.)
            })
            .collect();
        let output = graph.forward_tensor(view)?;
        close(&read(&output).await?, &expected)?;
        held.push((output, expected));
    }
    for (view, (_, expected)) in views.iter().zip(&held) {
        close(&read(&graph.forward_tensor_packed(view)?).await?, expected)?;
        graph.forward_tensor(view)?;
        graph.dispatch()?;
        close(&read(&graph.output_tensor()?).await?, expected)?;
    }
    let huge = device.upload(&[2, 3, 4], &[-f32::MAX; 24])?;
    let masked = huge
        .mul(&device.upload(&[1], &[2.])?)?
        .relu()?
        .narrow(2, 0, 3)?;
    let rejected = graph.forward_tensor(&masked)?;
    if read(&rejected).await.is_ok() {
        return Err("input view masked an inherited error".into());
    }
    graph.forward_tensor(&views[0])?;
    drop(graph);
    if read(&rejected).await.is_ok() {
        return Err("view guard lost after reuse/drop".into());
    }
    for (output, expected) in held {
        close(&read(&output).await?, &expected)?;
    }
    Ok(views.len() + 2)
}
