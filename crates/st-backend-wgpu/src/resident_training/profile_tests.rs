//! Diagnostic split-pass timestamps, not timing of the coalesced production path.
use super::*;
use crate::runtime::timestamps::{PassTimestampRecorder, TimestampErrorScopes};
use serde_json::json;

fn values(seed: &mut u32, len: usize, gain: f32) -> Vec<f32> {
    (0..len)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            ((*seed % 65) as f32 - 32.) / 64. * gain
        })
        .collect()
}

fn categories(depth: usize) -> Vec<&'static str> {
    let mut labels = vec!["forward"; depth];
    labels.extend(["loss_partials", "loss_reduce"]);
    for _ in 0..depth {
        labels.extend([
            "delta",
            "weight_gradient",
            "bias_gradient",
            "input_gradient",
        ]);
    }
    labels.extend(vec!["prepare_sgd"; depth]);
    labels.push("decide_sgd");
    labels.extend(vec!["commit_sgd"; depth]);
    labels
}

#[test]
#[ignore = "opt-in GPU diagnostic; splits Metal passes and perturbs scheduling"]
fn resident_training_dispatch_profile() {
    assert_eq!(
        std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref(),
        Ok("1")
    );
    // A private device prevents the error scopes from capturing another client.
    let runtime = WgpuRuntime::request_profiled_headless_blocking("training.profile").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    for (shape, depth) in [([2, 16, 32], 2), ([4, 16, 64], 8), ([4, 32, 128], 16)] {
        let layout = NdLayout::contiguous(&shape).unwrap();
        let width = shape[2];
        let mut seed = 17;
        let layers: Vec<_> = (0..depth)
            .map(|i| DenseLayer {
                inner: width,
                cols: width,
                weights: values(&mut seed, width * width, 1. / (width as f32).sqrt()),
                bias: values(&mut seed, width, 0.25),
                activation: if i + 1 == depth {
                    DenseActivation::None
                } else {
                    DenseActivation::Gelu
                },
            })
            .collect();
        let input = values(&mut seed, layout.len(), 1.);
        let target: Vec<_> = (0..input.len())
            .map(|i| 0.4 * input[i] - 0.2 * input[i / width * width + (i + 1) % width])
            .collect();
        let make = || {
            let mut gpu = ResidentDenseTraining::new(
                runtime.clone(),
                layout.clone(),
                &layers,
                MatmulTile::default(),
                MatmulKernel::Register2x2,
                MatmulAccumulation::Sequential,
            )
            .unwrap();
            gpu.upload_batch(&input, &target).unwrap();
            gpu
        };
        let mut gpu = make();
        let mut reference = make();
        let labels = categories(depth);
        assert_eq!(gpu.passes.len(), labels.len());
        let mut samples = Vec::new();
        for iteration in 0..10 {
            let context = runtime.context();
            let errors = TimestampErrorScopes::try_new(context.clone()).unwrap();
            let recorder =
                PassTimestampRecorder::new(context.clone(), labels.len() as u32).unwrap();
            context
                .queue()
                .write_buffer(&gpu.step_config, 0, bytemuck::bytes_of(&0.01f32));
            let mut encoder = context.device().create_command_encoder(&Default::default());
            encoder.clear_buffer(&gpu.validation, 0, None);
            for (i, pass) in gpu.passes.iter().enumerate() {
                let mut compute = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(labels[i]),
                    timestamp_writes: Some(recorder.writes(i as u32)),
                });
                pass.encode(&mut compute);
            }
            let mut readback = recorder.resolve(&mut encoder);
            context.queue().submit(Some(encoder.finish()));
            readback.validate(errors.finish());
            let timestamps = readback.read().unwrap();
            gpu.submitted_steps += 1;
            gpu.last_step = Some(gpu.submitted_steps);
            let loss = gpu.loss_snapshot().unwrap().read().unwrap();
            samples.push(
                json!({"iteration":iteration,"warmup":iteration < 2,"loss":loss,
                "period_ns":timestamps.timestamp_period_ns,
                "passes":timestamps.passes.iter().zip(&labels).map(|(p,label)| json!({
                    "operation":label,"start_tick":p.start_tick,"end_tick":p.end_tick,
                    "elapsed_ns":p.elapsed_ns})).collect::<Vec<_>>()}),
            );
            reference.step(0.01).unwrap();
            assert_eq!(loss, reference.loss_snapshot().unwrap().read().unwrap());
        }
        let actual = gpu.state_snapshot().unwrap().read().unwrap();
        let expected = reference.state_snapshot().unwrap().read().unwrap();
        assert_eq!(actual.loss, expected.loss);
        assert_eq!(actual.prediction, expected.prediction);
        assert_eq!(actual.input_gradient, expected.input_gradient);
        for (a, b) in actual.parameters.iter().zip(&expected.parameters) {
            assert_eq!(a.weights, b.weights);
            assert_eq!(a.bias, b.bias);
        }
        for (a, b) in actual
            .parameter_gradients
            .iter()
            .zip(&expected.parameter_gradients)
        {
            assert_eq!(a.weights, b.weights);
            assert_eq!(a.bias, b.bias);
        }
        println!(
            "TRAINING_PROFILE {}",
            json!({"schema":"spiraltorch.training_dispatch_profile.v1",
            "shape":shape,"depth":depth,"seed":17,"learning_rate":0.01,
            "adapter":format!("{:?}",runtime.adapter_info()),"samples":samples,
            "boundary":"private timestamp device; one compute pass per dispatch, including on Metal; changes scheduling and excludes host/readback cost; diagnostic only, not production throughput",
            "production_state_exact":true})
        );
    }
}
