//! Matched affine LayerNorm training with standalone and prepared resident graph routes.
#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use st_backend_wgpu::{
        resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulTile},
        resident_tensor::{pointwise::PointwisePlan, ResidentTensor, TensorDevice},
        resident_training::graph::ResidentGraphTraining,
        runtime,
    };
    use st_kernel_contracts::{
        graph::{GraphDefinition, GraphGradientPolicy, GraphParameter, GraphStage, ParameterRole},
        layout::NdLayout,
        pointwise::{PointwiseChain, PointwiseExecution, PointwiseStep},
    };
    use std::time::Instant;

    const STEPS: usize = 32;
    const WARMUP: usize = 2;
    const ITERATIONS: usize = 5;
    const EPSILON: f32 = 1e-5;
    const RATE: f32 = 0.01;

    fn values(n: usize, multiplier: usize, modulus: usize, divisor: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (((i * multiplier + 17) % modulus) as f32 - (modulus / 2) as f32) / divisor)
            .collect()
    }

    fn run_standalone(
        input: &ResidentTensor,
        target: &ResidentTensor,
        initial_gain: &ResidentTensor,
        initial_bias: &ResidentTensor,
        rate: &ResidentTensor,
        updates: &[PointwisePlan; 2],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let mut gain = initial_gain.clone();
        let mut bias = initial_bias.clone();
        let mut final_tensors = None;
        for _ in 0..STEPS {
            let tape = input.layer_norm_affine(&gain, &bias, EPSILON)?;
            let loss = tape.value().mean_squared_error(target)?;
            let [dx, dg, db] = tape.backward(loss.prediction_gradient(), 1., [true; 3])?;
            let dx = dx.unwrap();
            let dg = dg.unwrap();
            let db = db.unwrap();
            let next_gain = updates[0].run(&[&dg, rate, &gain], PointwiseExecution::Fused)?;
            let next_bias = updates[1].run(&[&db, rate, &bias], PointwiseExecution::Fused)?;
            final_tensors = Some((loss.value().clone(), tape.value().clone(), dx, dg, db));
            gain = next_gain;
            bias = next_bias;
        }
        let (loss, prediction, dx, dg, db) = final_tensors.unwrap();
        Ok(input
            .device()
            .snapshot_many(&[&loss, &prediction, &dx, &gain, &bias, &dg, &db])?
            .read()?)
    }

    let (runtime, _) = runtime::ensure_default_runtime_blocking("graph.layer_norm.benchmark")?;
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("non-CPU GPU adapter required".into());
    }
    let device = TensorDevice::new(runtime.clone())?;
    let mut cases = Vec::new();
    for (rows, cols) in [(2, 3), (32, 256), (128, 1025)] {
        let input_values = values(rows * cols, 37, 257, 64.);
        let target_values = values(rows * cols, 11, 67, 64.);
        let gain_values = vec![1.; cols];
        let bias_values = vec![0.; cols];
        let definition = GraphDefinition::new(
            NdLayout::contiguous(&[rows, cols])?,
            vec![GraphStage::LayerNorm {
                gain: 0,
                bias: 1,
                epsilon: EPSILON,
            }],
            vec![
                GraphParameter {
                    role: ParameterRole::Gain,
                    shape: vec![cols],
                    values: gain_values.clone(),
                },
                GraphParameter {
                    role: ParameterRole::Bias,
                    shape: vec![cols],
                    values: bias_values.clone(),
                },
            ],
        )?;
        let input = device.upload(&[rows, cols], &input_values)?;
        let target = device.upload(&[rows, cols], &target_values)?;
        let initial_gain = device.upload(&[cols], &gain_values)?;
        let initial_bias = device.upload(&[cols], &bias_values)?;
        let negative_rate = device.upload(&[], &[-RATE])?;
        let chain = PointwiseChain::new(
            3,
            vec![
                PointwiseStep::named("multiply", Some(1))?,
                PointwiseStep::named("add", Some(2))?,
            ],
        )?;
        let affine = NdLayout::contiguous(&[2 * cols])?;
        let gain_gradient = affine.narrow(0, 0, cols)?.reshape(&[cols])?;
        let bias_gradient = affine.narrow(0, cols, cols)?.reshape(&[cols])?;
        let updates = [
            PointwisePlan::new(
                device.clone(),
                chain.clone(),
                vec![
                    gain_gradient,
                    negative_rate.layout().clone(),
                    initial_gain.layout().clone(),
                ],
            )?,
            PointwisePlan::new(
                device.clone(),
                chain,
                vec![
                    bias_gradient,
                    negative_rate.layout().clone(),
                    initial_bias.layout().clone(),
                ],
            )?,
        ];
        let mut intervals = [Vec::new(), Vec::new()];
        let mut reference: Option<Vec<Vec<f32>>> = None;
        let mut max_scaled_error = 0f64;
        let mut terminal_loss = [0f32; 2];
        for iteration in 0..WARMUP + ITERATIONS {
            for j in 0..2 {
                let route = (iteration + j) % 2;
                let mut graph = if route == 1 {
                    let mut graph = ResidentGraphTraining::new(
                        runtime.clone(),
                        definition.clone(),
                        GraphGradientPolicy::Exact,
                        MatmulTile::default(),
                        MatmulKernel::Scalar,
                        MatmulAccumulation::Sequential,
                    )?;
                    graph.upload_batch(&input_values, &target_values)?;
                    Some(graph)
                } else {
                    None
                };
                let start = Instant::now();
                let output = if let Some(graph) = graph.as_mut() {
                    for _ in 0..STEPS {
                        graph.step(RATE)?;
                    }
                    let state = graph.state_snapshot()?.read()?;
                    vec![
                        vec![state.loss],
                        state.prediction,
                        state.input_gradient,
                        state.graph.parameters()[0].values.clone(),
                        state.graph.parameters()[1].values.clone(),
                        state.raw_gradients[0].clone(),
                        state.raw_gradients[1].clone(),
                    ]
                } else {
                    run_standalone(
                        &input,
                        &target,
                        &initial_gain,
                        &initial_bias,
                        &negative_rate,
                        &updates,
                    )?
                };
                let ms = start.elapsed().as_secs_f64() * 1000.;
                assert_eq!(output.len(), 7);
                assert!(output.iter().flatten().all(|value| value.is_finite()));
                terminal_loss[route] = output[0][0];
                if let Some(reference) = &reference {
                    for (actual, expected) in output.iter().zip(reference) {
                        assert_eq!(actual.len(), expected.len());
                        for (&actual, &expected) in actual.iter().zip(expected) {
                            let scaled = (f64::from(actual) - f64::from(expected)).abs()
                                / (5e-4 * (1. + f64::from(expected).abs()));
                            assert!(
                                scaled <= 1.,
                                "shape={rows}x{cols}, route={route}: {actual} != {expected}"
                            );
                            max_scaled_error = max_scaled_error.max(scaled);
                        }
                    }
                } else {
                    reference = Some(output);
                }
                if iteration >= WARMUP {
                    intervals[route].push(ms);
                }
            }
        }
        let mut sorted = intervals.clone();
        for times in &mut sorted {
            times.sort_by(f64::total_cmp);
        }
        cases.push(serde_json::json!({
            "rows": rows,
            "cols": cols,
            "standalone_ms": intervals[0],
            "graph_ms": intervals[1],
            "standalone_median_ms": sorted[0][ITERATIONS / 2],
            "graph_median_ms": sorted[1][ITERATIONS / 2],
            "max_scaled_error": max_scaled_error,
            "terminal_loss": terminal_loss,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "spiraltorch.layer_norm.resident_graph_matched.v1",
            "adapter": format!("{:?}", runtime.adapter_info()),
            "scope": "32-step affine LayerNorm + MSE + all VJPs + fused SGD; preloaded input and targets, preparation excluded, terminal read included",
            "routes": ["standalone_resident", "prepared_graph"],
            "steps": STEPS,
            "warmup": WARMUP,
            "iterations": ITERATIONS,
            "rate": RATE,
            "epsilon": EPSILON,
            "cases": cases,
        }))?
    );
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("native GPU benchmark only");
}
