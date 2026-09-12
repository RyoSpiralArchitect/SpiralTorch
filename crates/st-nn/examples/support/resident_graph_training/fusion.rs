//! The same fused/unfused trajectories and rejected updates in native and WASM.
use super::*;

pub(super) async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let mut cases = Vec::new();
    for (seed, shape) in [(17, vec![4]), (29, vec![3, 4]), (43, vec![2, 129, 4])] {
        for policy in [
            GraphGradientPolicy::Exact,
            GraphGradientPolicy::ModuleCompatible,
        ] {
            let mut reference = model(seed)?;
            let layout = NdLayout::contiguous(&shape)?;
            let rows = layout.len() / 4;
            let x = Tensor::from_fn(rows, 4, |r, c| ((r * 7 + c * 11) % 29) as f32 / 16. - 0.8)?;
            let y = Tensor::from_fn(rows, 3, |r, c| {
                0.4 * x.data()[r * 4 + c] - 0.2 * x.data()[r * 4 + 3] + c as f32 / 10.
            })?;
            let original = InferencePlan::from_module(&reference, layout)?;
            let plan = original.fuse_pointwise()?;
            if (original.stage_count(), plan.stage_count()) != (6, 5) {
                return Err("fixture no longer exercises pointwise stage fusion".into());
            }
            let initial = plan.graph_definition()?;
            let roles = initial
                .parameters()
                .iter()
                .map(|p| p.role)
                .collect::<Vec<_>>();
            let mut unfused = original.compile_graph_training_wgpu(runtime.clone(), policy)?;
            let mut fused = plan.compile_graph_training_wgpu(runtime.clone(), policy)?;
            unfused.upload_batch(x.data(), y.data())?;
            fused.upload_batch(x.data(), y.data())?;
            let rates = [0., 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125];
            let mut steps = Vec::new();
            let mut maximum = 0f32;
            let mut paired_maximum = 0f32;
            for rate in rates {
                unfused.step(rate)?;
                let before = state(unfused.state_snapshot()?).await?;
                fused.step(rate)?;
                let after = state(fused.state_snapshot()?).await?;
                let cpu = cpu_step(&mut reference, &x, &y, rate, policy, &roles)?;
                maximum = maximum
                    .max(compare(&before, &cpu)?)
                    .max(compare(&after, &cpu)?);
                paired_maximum = paired_maximum.max(compare(
                    &after,
                    &Reference {
                        loss: before.loss,
                        prediction: before.prediction,
                        dx: before.input_gradient,
                        raw: before.raw_gradients,
                        effective: before.effective_gradients,
                        parameters: before
                            .graph
                            .parameters()
                            .iter()
                            .map(|p| p.values.clone())
                            .collect(),
                    },
                )?);
                steps.push(state_json(&after));
            }
            let exported = InferencePlan::from_graph_definition(
                parameters(fused.parameter_snapshot()?).await?,
            )?;
            let restored = InferencePlan::from_json(&exported.to_json()?)?;
            let mut resumed = restored.compile_graph_training_wgpu(runtime.clone(), policy)?;
            resumed.upload_batch(x.data(), y.data())?;
            resumed.step(0.)?;
            maximum = maximum.max(compare(
                &state(resumed.state_snapshot()?).await?,
                &cpu_step(&mut reference, &x, &y, 0., policy, &roles)?,
            )?);
            cases.push(json!({"seed":seed,"input_shape":shape,"policy":format!("{policy:?}"),
                "plan":serde_json::from_str::<Value>(&plan.to_json()?)?,
                "source_plan":serde_json::from_str::<Value>(&original.to_json()?)?,
                "input":x.data(),"target":y.data(),"rates":rates,"steps":steps,
                "max_abs_error":maximum,"fused_unfused_max_abs_error":paired_maximum,"resume":"passed"}));
        }
    }
    Ok(json!({"cases":cases,"guards":guards(runtime).await?}))
}

async fn guards(runtime: WgpuRuntime) -> Result<Vec<Value>> {
    let mut records = Vec::new();
    for (name, rows, gains, relu, x, y, rate) in [
        ("masked_forward", 2, vec![f32::MAX], true, -2., 0., 0.01),
        (
            "masked_adjoint",
            1,
            vec![0., f32::MAX],
            false,
            0.,
            -1.,
            0.01,
        ),
        (
            "gain_unbroadcast",
            2,
            vec![0.],
            false,
            f32::MAX / 2.,
            -1.5,
            0.01,
        ),
        (
            "gain_candidate",
            1,
            vec![0.0625],
            true,
            16.,
            0.,
            f32::MAX / 8.,
        ),
    ] {
        let owned_parameters = gains
            .iter()
            .map(|&value| GraphParameter {
                role: ParameterRole::Gain,
                shape: vec![1],
                values: vec![value],
            })
            .collect();
        let mut stages: Vec<_> = gains
            .iter()
            .enumerate()
            .map(|(id, _)| GraphStage::Pointwise {
                chain: PointwiseChain::new(
                    2,
                    vec![PointwiseStep {
                        op: ElementwiseOp::Multiply,
                        rhs: Some(1),
                    }],
                )
                .unwrap(),
                parameters: vec![id],
            })
            .collect();
        stages.push(GraphStage::Pointwise {
            chain: PointwiseChain::new(
                1,
                vec![PointwiseStep {
                    op: if relu {
                        ElementwiseOp::Relu
                    } else {
                        ElementwiseOp::Identity
                    },
                    rhs: None,
                }],
            )?,
            parameters: vec![],
        });
        let source = InferencePlan::from_graph_definition(GraphDefinition::new(
            NdLayout::contiguous(&[rows, 1, 1])?,
            stages,
            owned_parameters,
        )?)?;
        for policy in [
            GraphGradientPolicy::Exact,
            GraphGradientPolicy::ModuleCompatible,
        ] {
            for optimized in [false, true] {
                let plan = if optimized {
                    source.fuse_pointwise()?
                } else {
                    source.clone()
                };
                if optimized && plan.stage_count() != 1 {
                    return Err("guard was not fused".into());
                }
                let initial = plan.graph_definition()?;
                let mut gpu = plan.compile_graph_training_wgpu(runtime.clone(), policy)?;
                gpu.upload_batch(&vec![x; rows], &vec![y; rows])?;
                gpu.step(rate)?;
                let failed = gpu.loss_snapshot()?;
                let failed_state = gpu.state_snapshot()?;
                let prediction = gpu.prediction_tensor()?;
                let gradient = gpu.input_gradient_tensor()?;
                let rollback = gpu.parameter_snapshot()?;
                gpu.upload_batch(&vec![0.; rows], &vec![0.; rows])?;
                gpu.step(0.01)?;
                let recovered = gpu.state_snapshot()?;
                drop(gpu);
                let error = loss(failed)
                    .await
                    .expect_err("fused guard accepted overflow");
                if !matches!(
                    error.downcast_ref::<TrainingError>(),
                    Some(TrainingError::Rejected { .. })
                ) || state(failed_state).await.is_ok()
                    || tensor(prediction.snapshot()?).await.is_ok()
                    || tensor(gradient.snapshot()?).await.is_ok()
                {
                    return Err("fused guard or owning snapshot lost rejection".into());
                }
                let recovered = state(recovered).await?;
                if bits(&parameters(rollback).await?) != bits(&initial)
                    || bits(&recovered.graph) != bits(&initial)
                    || recovered.loss != 0.
                {
                    return Err("fused rejection partially committed or failed to recover".into());
                }
                records.push(json!({"case":name,"optimized":optimized,"policy":format!("{policy:?}"),
                    "error":error.to_string(),"all_parameter_bits_unchanged":true,"recovered":true}));
            }
        }
    }
    Ok(records)
}
