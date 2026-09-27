#[cfg(target_arch = "wasm32")]
mod browser {
    use st_backend_wgpu::{resident_tensor::TensorDevice, runtime::WgpuRuntime};
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn close(actual: &[f32], expected: &[f32]) -> Result<()> {
        if actual.len() != expected.len() {
            return Err("depthwise VJP output length".into());
        }
        for (&value, &reference) in actual.iter().zip(expected) {
            if !value.is_finite() || (value - reference).abs() > 2e-5 * (1.0 + reference.abs()) {
                return Err(format!("depthwise VJP: {value} != {reference}").into());
            }
        }
        Ok(())
    }

    pub async fn run() -> Result<String> {
        let runtime = WgpuRuntime::request_headless("depthwise_vjp.browser").await?;
        let device = TensorDevice::new(runtime.clone())?;
        let input_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let weight_values = [1.0, -0.5, 0.25, 2.0];
        let upstream_values = [1.0, 0.5, -1.0, 2.0];
        let input = device.upload(&[1, 1, 3, 3], &input_values)?;
        let weights = device.upload(&[1, 2, 2], &weight_values)?;
        let upstream = device.upload(&[1, 1, 2, 2], &upstream_values)?;
        let gradients = input.depthwise_conv2d_vjp(&weights, &upstream, (1, 1), (0, 0), (1, 1))?;
        let mut expected_input = [0.0; 9];
        let mut expected_weight = [0.0; 4];
        let mut expected_bias = [0.0; 1];
        for oy in 0..2 {
            for ox in 0..2 {
                let seed = upstream_values[oy * 2 + ox];
                expected_bias[0] += seed;
                for ky in 0..2 {
                    for kx in 0..2 {
                        let input_index = (oy + ky) * 3 + ox + kx;
                        let weight_index = ky * 2 + kx;
                        expected_input[input_index] += seed * weight_values[weight_index];
                        expected_weight[weight_index] += seed * input_values[input_index];
                    }
                }
            }
        }
        for (gradient, expected) in gradients.iter().zip([
            expected_input.as_slice(),
            expected_weight.as_slice(),
            expected_bias.as_slice(),
        ]) {
            close(&gradient.snapshot()?.read_async().await?, expected)?;
        }

        let empty_input = device.upload(&[0, 1, 2, 2], &[])?;
        let unit_weight = device.upload(&[1, 1, 1], &[1.0])?;
        let empty_upstream = device.upload(&[0, 1, 2, 2], &[])?;
        let empty = empty_input.depthwise_conv2d_vjp(
            &unit_weight,
            &empty_upstream,
            (1, 1),
            (0, 0),
            (1, 1),
        )?;
        close(&empty[0].snapshot()?.read_async().await?, &[])?;
        close(&empty[1].snapshot()?.read_async().await?, &[0.0])?;
        close(&empty[2].snapshot()?.read_async().await?, &[0.0])?;

        let overflowing = device
            .upload(&[1, 1, 1, 1], &[f32::MAX])?
            .depthwise_conv2d_vjp(
                &unit_weight,
                &device.upload(&[1, 1, 1, 1], &[2.0])?,
                (1, 1),
                (0, 0),
                (1, 1),
            )?;
        for gradient in overflowing {
            if gradient.snapshot()?.read_async().await.is_ok() {
                return Err("depthwise VJP lost a shared overflow guard".into());
            }
        }

        Ok(serde_json::to_string(&serde_json::json!({
            "schema": "spiraltorch.resident_depthwise_vjp.browser.v1",
            "status": "passed",
            "adapter": format!("{:?}", runtime.adapter_info()),
            "gradient_cases": 3,
            "empty_batch_checks": 3,
            "shared_guard_checks": 3,
            "intermediate_readbacks": 0,
        }))?)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_depthwise_vjp_checks() -> Result<String, wasm_bindgen::JsValue> {
    browser::run()
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
