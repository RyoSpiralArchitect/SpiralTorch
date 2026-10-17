use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use st_backend_wgpu::{
    resident_tensor::{attention::AttentionMask, ResidentTensor, TensorDevice},
    runtime::{self, WgpuRuntime},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

async fn read(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = tensor.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    Ok(values)
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let fixture_raw = include_str!("../fixtures/wide_rounding.json");
    let fixture: Value = serde_json::from_str(fixture_raw)?;
    if fixture["schema"] != "spiraltorch.wide_rounding.fixture.v1" {
        return Err("unexpected fixture schema".into());
    }
    let cases: Vec<[i64; 5]> = serde_json::from_value(fixture["cases"].clone())?;
    if !(1000..=8192).contains(&cases.len()) {
        return Err("unexpected fixture size".into());
    }
    let mut input = Vec::with_capacity(cases.len() * 4);
    let mut expected = Vec::with_capacity(cases.len());
    for case in &cases {
        for &bits in &case[..3] {
            input.push(u32::try_from(bits)?);
        }
        input.push(i32::try_from(case[3])? as u32);
        expected.push(u32::try_from(case[4])?);
    }
    let source = [
        include_str!("../../src/shaders/rounded_add.wgsl"),
        include_str!("../../src/resident_tensor/shaders/layer_norm_wide.wgsl"),
        "@group(0) @binding(0) var<storage, read> input: array<u32>;\n\
         @group(0) @binding(1) var<storage, read_write> output: array<u32>;\n\
         @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {\n\
           if (id.x >= arrayLength(&output)) { return; }\n\
           let p = id.x * 4u;\n\
           output[id.x] = bitcast<u32>(wide_float(Wide(bitcast<f32>(input[p]),\n\
               bitcast<f32>(input[p+1u]), bitcast<i32>(input[p+3u]), bitcast<f32>(input[p+2u]))));\n\
         }",
    ].join("\n");
    let context = runtime.context();
    let device = context.device();
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let input_buffer =
        runtime::upload_slice(device, "wide.inputs", &input, wgpu::BufferUsages::STORAGE)?;
    let output_buffer = runtime::empty_buffer::<u32>(
        device,
        "wide.outputs",
        cases.len(),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    )?;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("wide.rounding"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("wide.rounding"),
        layout: None,
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("wide.rounding"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("wide.rounding"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
    }
    context.queue().submit(Some(encoder.finish()));
    if let Some(error) = device.pop_error_scope().await {
        return Err(error.into());
    }
    #[cfg(not(target_arch = "wasm32"))]
    let actual: Vec<u32> = runtime::read_buffer(
        device,
        context.queue(),
        &output_buffer,
        cases.len(),
        "wide.read",
    )?;
    #[cfg(target_arch = "wasm32")]
    let actual: Vec<u32> =
        runtime::read_buffer_async(context.clone(), &output_buffer, cases.len(), "wide.read")
            .await?;
    let mismatches: Vec<_> = expected
        .iter()
        .zip(&actual)
        .enumerate()
        .filter_map(|(index, (expected, actual))| (expected != actual).then_some(index))
        .collect();
    let tensor_device = TensorDevice::new(runtime.clone())?;
    let query = tensor_device.upload(&[1, 1, 1, 1], &[0.])?;
    let keys = tensor_device.upload(&[1, 1, 4, 1], &[0.; 4])?;
    let mut attention = Vec::new();
    for sign in [0u32, 0x8000_0000] {
        for high in [0x4080_0000u32, 0x4080_0001] {
            for tail_sign in [-1, 0, 1] {
                let values = [
                    f32::from_bits(high),
                    2f32.powi(-22),
                    tail_sign as f32 * 2f32.powi(-47),
                    0.,
                ];
                let values = values.map(|x| f32::from_bits(x.to_bits() ^ sign));
                let values = tensor_device.upload(&[1, 1, 4, 1], &values)?;
                let result = query.scaled_dot_attention(
                    &keys,
                    &values,
                    1.,
                    AttentionMask::None,
                    None,
                    None,
                )?;
                let actual = read(&result).await?[0].to_bits();
                let increment = u32::from(tail_sign > 0 || (tail_sign == 0 && high & 1 != 0));
                let expected = ((high - (2 << 23)) + increment) ^ sign;
                attention.push(json!({"sign":sign,"high":high,"tail_sign":tail_sign,
                                      "expected_bits":expected,"actual_bits":actual,"passed":actual == expected}));
            }
        }
    }
    Ok(json!({"schema":"spiraltorch.wide_rounding.validation.v1",
        "fixture_sha256":format!("{:x}", Sha256::digest(fixture_raw.as_bytes())),
        "backend":format!("{:?}", runtime.adapter_info().backend),
        "device_type":format!("{:?}", runtime.adapter_info().device_type),
        "browser_hardware_execution":if cfg!(target_arch="wasm32") { "unverified" } else { "not_applicable" },
        "case_count":cases.len(),"actual_bits":actual,"mismatch_indices":mismatches,
        "passed":mismatches.is_empty() && attention.iter().all(|c| c["passed"] == true),
        "attention":attention}))
}
