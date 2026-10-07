//! Bounded correctness probe: all 100 updates precede the first host readback.
//! Input/target uploads and allocation remain; this is not a throughput claim.

#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("run the native probe; browser execution needs async readback");
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;
    use st_backend_wgpu::{
        resident_tensor::{
            pointwise::{vjp::PointwiseVjpPlan, PointwisePlan},
            TensorDevice,
        },
        runtime,
    };
    use st_kernel_contracts::{
        elementwise::ElementwiseOp, layout::NdLayout, pointwise::PointwiseExecution,
        topos_resonator::ToposResonatorKernel,
    };
    use std::{fs::OpenOptions, io::Write};

    let args: Vec<_> = std::env::args().collect();
    if args.len() != 2 {
        return Err("usage: topos_resident_learning NEW_JSON".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    let (runtime, _) = runtime::ensure_default_runtime_blocking("topos.resident.learning")?;
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("probe requires hardware GPU, not a CPU adapter".into());
    }
    let adapter = format!("{:?}", runtime.adapter_info());
    let device = TensorDevice::new(runtime)?;
    let features = 5;
    let rate = 0.03_f32;
    let rate_tensor = device.upload(&[], &[rate])?;
    let initial = [0.7_f32, -0.8, 0.4, 1.2, -1.1];
    let mut cases = Vec::new();
    for porosity in [0.0_f32, 0.3] {
        let kernel = ToposResonatorKernel::new(0.2, 1., porosity, 5)?;
        let mut gate = device.upload(&[1, features], &initial)?;
        let mut plans = Vec::new();
        for (index, rows) in [1, 3, 8, 2].into_iter().enumerate() {
            let layout = if index % 2 == 0 {
                NdLayout::contiguous(&[rows, features])?
            } else {
                NdLayout::contiguous(&[features, rows])?.permute(&[1, 0])?
            };
            plans.push(PointwiseVjpPlan::new(PointwisePlan::topos_resonator(
                device.clone(),
                kernel,
                vec![layout, gate.layout().clone()],
            )?)?);
        }
        let mut records = Vec::new();
        let mut retained = Vec::new();
        for step in 0..100 {
            let rows = [1, 3, 8, 2][step % 4];
            let x: Vec<_> = (0..rows * features)
                .map(|i| {
                    ((i / features * 13 + i % features * 7 + step * 3) % 29) as f32 * 0.11 - 1.5
                })
                .collect();
            let target: Vec<_> = (0..rows * features)
                .map(|i| ((i / features * 5 + i % features * 11) % 23) as f32 * 0.02 - 0.22)
                .collect();
            let input = if step % 2 == 0 {
                device.upload(&[rows, features], &x)?
            } else {
                let column_major: Vec<_> = (0..features)
                    .flat_map(|c| (0..rows).map(move |r| r * features + c))
                    .map(|i| x[i])
                    .collect();
                device
                    .upload(&[features, rows], &column_major)?
                    .permute(&[1, 0])?
            };
            let target_tensor = device.upload(&[rows, features], &target)?;
            let plan = &plans[step % 4];
            let output = plan
                .forward()
                .run(&[&input, &gate], PointwiseExecution::Fused)?;
            let loss = output.mean_squared_error(&target_tensor)?;
            let gradients = plan.run(&[&input, &gate], loss.prediction_gradient())?;
            let delta = gradients[1].apply(ElementwiseOp::Multiply, Some(&rate_tensor))?;
            gate = gate.apply(ElementwiseOp::Subtract, Some(&delta))?;
            // Owning immutable tensors, not readback bytes or weights fed back from CPU.
            retained.push([
                output,
                gradients[0].clone(),
                gradients[1].clone(),
                gate.clone(),
                loss.value().clone(),
            ]);
            records.push(
                json!({"step":step, "shape":[rows,features], "input":x, "target":target,
                "input_layout": if step % 2 == 0 {"row_major"} else {"col_major"}}),
            );
        }
        let tensors: Vec<_> = retained.iter().flatten().collect();
        let snapshots = device.snapshot_many(&tensors)?.read()?;
        for (record, values) in records.iter_mut().zip(snapshots.as_chunks::<5>().0) {
            for (name, data) in ["output", "grad_input", "grad_gate", "gate_after"]
                .into_iter()
                .zip(values)
            {
                record[name] = json!(data);
            }
            record["loss"] = json!(values[4][0]);
        }
        cases.push(json!({"config":{"coupling":kernel.coupling(),"iterations":kernel.iterations(),"saturation":kernel.saturation(),"porosity":kernel.porosity()},"initial_gate":initial,"records":records}));
    }
    serde_json::to_writer_pretty(
        &mut file,
        &json!({
            "schema":"spiraltorch.topos_resident_learning.v1", "status":"executed",
            "backend":"wgpu", "adapter":adapter, "dtype":"float32", "learning_rate":rate,
            "optimizer":"resident_subtract_lr_times_gradient", "gate_layout":"shared_rows",
            "gate_gradient_reduction":"sum_without_additional_mean", "host_readback_during_updates":false,
            "cases":cases,
            "scope":"Two synthetic 100-update trajectories: Topos, MSE, VJP, gate sum and immutable SGD stay resident. Input/target uploads and per-step allocations remain. This is not transactional graph SGD, a ModuleTrainer/Sequential integration, a browser execution, or a timing/model-quality claim."
        }),
    )?;
    file.write_all(b"\n")?;
    println!(
        "executed 200 resident Topos updates; readbacks only after each 100-update trajectory"
    );
    Ok(())
}
