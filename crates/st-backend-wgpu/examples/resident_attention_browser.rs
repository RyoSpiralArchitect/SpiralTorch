//! Thin browser probe of the same resident kernel and frozen PyTorch oracle.

#[cfg(target_arch = "wasm32")]
mod browser {
    use st_backend_wgpu::{
        resident_tensor::{attention::AttentionMask, ResidentTensor, TensorDevice},
        runtime::WgpuRuntime,
    };
    use st_kernel_contracts::{attention::AttentionSpec, layout::NdLayout};

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    async fn check_compute_constants(runtime: &WgpuRuntime) -> Result<Vec<serde_json::Value>> {
        let context = runtime.context();
        let gpu = context.device();
        gpu.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention.constant_probe"),
            source: wgpu::ShaderSource::Wgsl(
                "override VALUE: f32 = 1.0;
                 override COUNT: u32 = 1u;
                 @id(7) override SHIFT: u32 = 0u;
                 @group(0) @binding(0) var<storage, read_write> output: array<f32>;
                 @compute @workgroup_size(1) fn main() {
                     output[0] = VALUE + f32(COUNT) + f32(SHIFT);
                 }"
                .into(),
            ),
        });
        let mut checks = Vec::new();
        for (name, constants, expected) in [
            ("defaults", std::collections::HashMap::new(), 2f32),
            (
                "named_and_numeric",
                std::collections::HashMap::from([
                    ("VALUE".to_owned(), 3.5),
                    ("COUNT".to_owned(), 4.),
                    ("7".to_owned(), 7.),
                ]),
                14.5,
            ),
            (
                "second_pipeline",
                std::collections::HashMap::from([
                    ("VALUE".to_owned(), -2.5),
                    ("COUNT".to_owned(), 5.),
                    ("7".to_owned(), 11.),
                ]),
                13.5,
            ),
        ] {
            let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
            });
            let output = gpu.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let binding = gpu.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(name),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output.as_entire_binding(),
                }],
            });
            let mut encoder = gpu.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &binding, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            context.queue().submit(Some(encoder.finish()));
            let actual = st_backend_wgpu::runtime::read_buffer_async::<f32>(
                context.clone(),
                &output,
                1,
                "constant_probe",
            )
            .await?;
            if actual != [expected] {
                return Err(format!("compute constants {name}: {actual:?} != {expected}").into());
            }
            checks
                .push(serde_json::json!({"name": name, "actual": actual[0], "expected": expected}));
        }
        if let Some(error) = gpu.pop_error_scope().await {
            return Err(format!("compute constants WebGPU validation: {error}").into());
        }
        Ok(checks)
    }

    // Materialize the same oracle values into a reversed, padded storage layout.
    // Only this test setup rearranges host data; attention must read the view.
    fn upload_view(
        device: &TensorDevice,
        shape: &[usize],
        values: &[f32],
        strided: bool,
    ) -> Result<ResidentTensor> {
        if !strided {
            return Ok(device.upload(shape, values)?);
        }
        let axes: Vec<_> = (0..shape.len()).rev().collect();
        let mut storage_shape: Vec<_> = shape.iter().rev().copied().collect();
        storage_shape[0] += 2;
        let storage = NdLayout::contiguous(&storage_shape)?;
        let view = storage
            .narrow(0, 1, shape[shape.len() - 1])?
            .permute(&axes)?;
        if values.len() != view.len() {
            return Err("fixture length".into());
        }
        let mut data = vec![0.; storage.len()];
        for (i, &value) in values.iter().enumerate() {
            data[view.storage_index(i).ok_or("fixture length")?] = value;
        }
        Ok(device
            .upload(&storage_shape, &data)?
            .narrow(0, 1, shape[shape.len() - 1])?
            .permute(&axes)?)
    }

    pub async fn run() -> Result<String> {
        let runtime = WgpuRuntime::request_headless("resident.attention.browser").await?;
        let device = TensorDevice::new(runtime.clone())?;
        let constant_checks = check_compute_constants(&runtime).await?;
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/resident_attention_torch.json"
        ))?;
        let floats = |v: &serde_json::Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let shape = |v: &serde_json::Value| -> Vec<usize> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect()
        };
        let mut checks = Vec::new();
        for case in fixture["cases"].as_array().unwrap() {
            let q_shape = shape(&case["query_shape"]);
            let k_shape = shape(&case["key_shape"]);
            let scale = case["scale"].as_f64().unwrap() as f32;
            let mask = case["query_offset"]
                .as_u64()
                .map_or(AttentionMask::None, |offset| AttentionMask::Causal {
                    query_offset: offset as usize,
                });
            let spec = AttentionSpec::new(&q_shape, &k_shape, &k_shape, scale, mask)?;
            for (strided, merged) in [(false, false), (true, false), (false, true), (true, true)] {
                let q = upload_view(&device, &q_shape, &floats(&case["query"]), strided)?;
                let k = upload_view(&device, &k_shape, &floats(&case["key"]), strided)?;
                let v = upload_view(&device, &k_shape, &floats(&case["value"]), strided)?;
                let z = if case["z_bias"].is_null() {
                    None
                } else {
                    Some(upload_view(
                        &device,
                        &spec.z_bias_shape(),
                        &floats(&case["z_bias"]),
                        strided,
                    )?)
                };
                let pair = if case["pair_bias"].is_null() {
                    None
                } else {
                    Some(upload_view(
                        &device,
                        &spec.pair_bias_shape(),
                        &floats(&case["pair_bias"]),
                        strided,
                    )?)
                };
                runtime
                    .context()
                    .device()
                    .push_error_scope(wgpu::ErrorFilter::Validation);
                let output = if merged {
                    q.scaled_dot_attention_merged_heads(
                        &k,
                        &v,
                        scale,
                        mask,
                        z.as_ref(),
                        pair.as_ref(),
                    )?
                } else {
                    q.scaled_dot_attention(&k, &v, scale, mask, z.as_ref(), pair.as_ref())?
                };
                if let Some(error) = runtime.context().device().pop_error_scope().await {
                    return Err(format!("{}: WebGPU validation: {error}", case["name"]).into());
                }
                drop((q, k, v, z, pair));
                let actual = output.snapshot()?.read_async().await?;
                let mut expected = floats(&case["expected"]);
                let expected_shape = if merged {
                    let view = NdLayout::contiguous(&q_shape)?.permute(&[0, 2, 1, 3])?;
                    expected = (0..view.len())
                        .map(|i| expected[view.storage_index(i).unwrap()])
                        .collect();
                    spec.merged_output_shape()?.to_vec()
                } else {
                    q_shape.clone()
                };
                if output.layout().shape() != expected_shape
                    || !output.layout().is_contiguous()
                    || output.layout().offset() != 0
                {
                    return Err("attention output layout".into());
                }
                if actual.len() != expected.len() {
                    return Err("attention output length".into());
                }
                let mut max_error = 0f32;
                for (&a, &b) in actual.iter().zip(&expected) {
                    if !a.is_finite() || !b.is_finite() || (a - b).abs() > 3e-6 + 3e-5 * b.abs() {
                        return Err(format!("{}: {a} != {b}", case["name"]).into());
                    }
                    max_error = max_error.max((a - b).abs());
                }
                checks.push(serde_json::json!({"name": case["name"], "layout": if strided {"strided"} else {"canonical"}, "output_order": if merged {"merged_heads"} else {"head_major"}, "max_abs_error": max_error}));
            }
        }
        if checks.len() != 80 {
            return Err("incomplete attention fixture".into());
        }
        Ok(serde_json::to_string(&serde_json::json!({
            "schema": "spiraltorch.resident_attention.browser.v1",
            "passed": true,
            "adapter": format!("{:?}", runtime.adapter_info()),
            "reference_torch_version": fixture["torch_version"],
            "checks": checks,
            "constant_checks": constant_checks,
            "scope": "forward numerical parity only; no speed or learning claim",
        }))?)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_resident_attention_checks() -> Result<String, wasm_bindgen::JsValue> {
    browser::run()
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
