#[cfg(target_arch = "wasm32")]
mod browser {
    use st_backend_wgpu::{resident_tensor::TensorDevice, runtime::WgpuRuntime};
    use st_core::backend::device_caps::DeviceCaps;
    use st_nn::{
        execution::{push_backend_policy, BackendPolicy},
        module::Module,
    };
    use st_tensor::Tensor;
    use st_vision::models::{ConvNeXtBackbone, ConvNeXtConfig};

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    const TOLERANCE: f32 = 1e-2;

    fn close(actual: &[f32], expected: &[f32]) -> Result<[f32; 2]> {
        if actual.len() != expected.len() {
            return Err("ConvNeXt backbone VJP length mismatch".into());
        }
        let mut errors = [0.0_f32; 2];
        for (&value, &reference) in actual.iter().zip(expected) {
            let absolute = (value - reference).abs();
            let scaled = absolute / (1.0 + reference.abs());
            if !value.is_finite() || !reference.is_finite() || scaled > TOLERANCE {
                return Err(format!("ConvNeXt backbone VJP: {value} != {reference}").into());
            }
            errors[0] = errors[0].max(absolute);
            errors[1] = errors[1].max(scaled);
        }
        Ok(errors)
    }

    pub async fn run() -> Result<String> {
        let runtime = WgpuRuntime::request_headless("vision.convnext_backbone_vjp.browser").await?;
        let device = TensorDevice::new(runtime)?;
        let mut backbone = ConvNeXtBackbone::new(ConvNeXtConfig {
            input_channels: 2,
            input_hw: (8, 8),
            stage_dims: vec![3, 4],
            stage_depths: vec![1, 1],
            patch_size: (2, 2),
            curvature: -1.0,
            epsilon: 1e-6,
        })?;
        let input = Tensor::from_fn(2, 128, |row, col| {
            ((row * 41 + col * 17) % 101) as f32 / 101.0 - 0.5
        })?;
        let seed = Tensor::from_fn(2, 16, |row, col| {
            ((row * 13 + col * 7) % 37) as f32 / 37.0 - 0.4
        })?;
        let resident_input = device.upload(&[2, 2, 8, 8], input.data())?;
        let resident_seed = device.upload(&[2, 16], seed.data())?;
        let gradients = backbone.vjp_resident(&resident_input, &resident_seed)?;
        let expected_input = {
            let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
            backbone.backward(&input, &seed)?
        };
        let mut errors = close(
            &gradients.input_gradient().snapshot()?.read_async().await?,
            expected_input.data(),
        )?;
        let mut references = Vec::new();
        backbone.visit_parameters(&mut |parameter| {
            references.push(parameter.gradient().unwrap().data().to_vec());
            Ok(())
        })?;
        if gradients.parameter_gradients().len() != 22 || references.len() != 22 {
            return Err("ConvNeXt backbone VJP parameter count".into());
        }
        for (gradient, expected) in gradients.parameter_gradients().iter().zip(&references) {
            let next = close(&gradient.snapshot()?.read_async().await?, expected)?;
            errors[0] = errors[0].max(next[0]);
            errors[1] = errors[1].max(next[1]);
        }
        Ok(format!(
            "{{\"schema\":\"spiraltorch.convnext_backbone_vjp.browser.v1\",\"status\":\"passed\",\"gradient_tensors\":23,\"intermediate_readbacks\":0,\"max_absolute_error\":{},\"max_scaled_error\":{},\"scaled_error_tolerance\":{TOLERANCE}}}",
            errors[0], errors[1]
        ))
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_convnext_backbone_vjp_checks() -> Result<String, wasm_bindgen::JsValue> {
    browser::run()
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
