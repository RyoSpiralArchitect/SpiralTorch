//! Thin browser probe of the same resident kernel and frozen PyTorch oracle.

#[cfg(target_arch = "wasm32")]
mod browser {
    use st_backend_wgpu::{
        resident_tensor::{attention::AttentionMask, TensorDevice},
        runtime::WgpuRuntime,
    };
    use st_kernel_contracts::attention::AttentionSpec;

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    pub async fn run() -> Result<String> {
        let runtime = WgpuRuntime::request_headless("resident.attention.browser").await?;
        let device = TensorDevice::new(runtime.clone())?;
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
            let q = device.upload(&q_shape, &floats(&case["query"]))?;
            let k = device.upload(&k_shape, &floats(&case["key"]))?;
            let v = device.upload(&k_shape, &floats(&case["value"]))?;
            let z = if case["z_bias"].is_null() {
                None
            } else {
                Some(device.upload(&spec.z_bias_shape(), &floats(&case["z_bias"]))?)
            };
            let pair = if case["pair_bias"].is_null() {
                None
            } else {
                Some(device.upload(&spec.pair_bias_shape(), &floats(&case["pair_bias"]))?)
            };
            runtime
                .context()
                .device()
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let output = q.scaled_dot_attention(&k, &v, scale, mask, z.as_ref(), pair.as_ref())?;
            if let Some(error) = runtime.context().device().pop_error_scope().await {
                return Err(format!("{}: WebGPU validation: {error}", case["name"]).into());
            }
            drop((q, k, v, z, pair));
            let actual = output.snapshot()?.read_async().await?;
            let expected = floats(&case["expected"]);
            if actual.len() != expected.len() {
                return Err("attention output length".into());
            }
            let mut max_error = 0f32;
            for (&a, &b) in actual.iter().zip(&expected) {
                if !a.is_finite() || (a - b).abs() > 3e-6 + 3e-5 * b.abs() {
                    return Err(format!("{}: {a} != {b}", case["name"]).into());
                }
                max_error = max_error.max((a - b).abs());
            }
            checks.push(serde_json::json!({"name": case["name"], "max_abs_error": max_error}));
        }
        if checks.len() != 20 {
            return Err("incomplete attention fixture".into());
        }
        Ok(serde_json::to_string(&serde_json::json!({
            "schema": "spiraltorch.resident_attention.browser.v1",
            "passed": true,
            "adapter": format!("{:?}", runtime.adapter_info()),
            "reference_torch_version": fixture["torch_version"],
            "checks": checks,
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
