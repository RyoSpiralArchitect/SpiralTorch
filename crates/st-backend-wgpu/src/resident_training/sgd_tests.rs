use super::*;
use st_kernel_contracts::sgd::{sgd_candidate_wgsl, SgdError, SGD_INVALID_GRADIENT};

#[test]
fn shared_sgd_candidate_matches_cpu_for_zero_rate_and_overflow() {
    if std::env::var("SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS").as_deref() != Ok("1") {
        return;
    }
    let (runtime, _) = runtime::ensure_default_runtime_blocking("training.sgd.contract").unwrap();
    assert_ne!(runtime.adapter_info().device_type, wgpu::DeviceType::Cpu);
    let context = runtime.context();
    let device = context.device();
    let cases = [
        [4.0, 8.0, 0.25, 0.0],
        [-2.0, -4.0, 0.25, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [-0.0, 1.0, 0.0, 0.0],
        [f32::MAX, f32::MAX, 0.0, 0.0],
        [f32::MAX, f32::MAX, 1.0, 0.0],
        [f32::MAX, f32::MAX, 2.0, 0.0],
        [f32::MAX, -f32::MAX, 1.0, 0.0],
        [1.0, f32::NAN, 0.0, 0.0],
        [1.0, f32::INFINITY, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0, 0.0],
        [1.0, f32::NEG_INFINITY, 0.5, 0.0],
    ];
    let source = sgd_candidate_wgsl()
        + r#"
@group(0) @binding(0) var<storage, read> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> results: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let input = inputs[id.x];
    let plain = sgd_candidate(input.x, input.y, input.z, 4096u);
    let momentum = sgd_candidate(input.x, input.y, input.z, 32768u);
    results[id.x * 4u] = bitcast<u32>(plain.value);
    results[id.x * 4u + 1u] = plain.flags;
    results[id.x * 4u + 2u] = bitcast<u32>(momentum.value);
    results[id.x * 4u + 3u] = momentum.flags;
}
"#;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sgd.contract.shader"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("sgd.contract.pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
    });
    let input =
        runtime::upload_slice(device, "sgd.cases", &cases, wgpu::BufferUsages::STORAGE).unwrap();
    let output = runtime::empty_buffer::<u32>(
        device,
        "sgd.results",
        cases.len() * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )
    .unwrap();
    let binding = binding(
        device,
        &pipeline.get_bind_group_layout(0),
        &[&input, &output],
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &binding, &[]);
        pass.dispatch_workgroups(cases.len() as u32, 1, 1);
    }
    context.queue().submit(Some(encoder.finish()));
    let bytes = readback::capture_new(context, &[&output])
        .unwrap()
        .read()
        .unwrap();
    let (rows, remainder) = bytes.as_chunks::<16>();
    assert!(remainder.is_empty());
    assert_eq!(rows.len(), cases.len());
    for (case, bytes) in cases.iter().zip(rows) {
        let flags = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let momentum_flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        match SgdStep::new(case[2]).unwrap().candidate(case[0], case[1]) {
            Ok(expected) => {
                assert_eq!(flags, 0, "{case:?}");
                assert_eq!(momentum_flags, 0, "{case:?}");
                assert_eq!(f32::from_le_bytes(bytes[..4].try_into().unwrap()), expected);
                assert_eq!(
                    f32::from_le_bytes(bytes[8..12].try_into().unwrap()),
                    expected
                );
            }
            Err(SgdError::NonFinite { flags: expected }) => {
                assert_eq!(flags, expected, "{case:?}");
                let gradient_flag = if expected & SGD_INVALID_GRADIENT != 0 {
                    32768
                } else {
                    0
                };
                assert_eq!(
                    momentum_flags,
                    (expected & !SGD_INVALID_GRADIENT) | gradient_flag,
                    "{case:?}"
                );
            }
            Err(error) => panic!("unexpected CPU oracle failure: {error}"),
        }
    }
}
