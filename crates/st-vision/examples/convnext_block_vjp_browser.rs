#[cfg(target_arch = "wasm32")]
mod browser {
    use st_backend_wgpu::{resident_tensor::TensorDevice, runtime::WgpuRuntime};
    use st_core::backend::device_caps::DeviceCaps;
    use st_nn::{
        execution::{push_backend_policy, BackendPolicy},
        module::Module,
    };
    use st_tensor::Tensor;
    use st_vision::models::ConvNeXtBlock;

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn close(actual: &[f32], expected: &[f32]) -> Result<()> {
        if actual.len() != expected.len() {
            return Err("ConvNeXt VJP length mismatch".into());
        }
        for (&value, &reference) in actual.iter().zip(expected) {
            if !value.is_finite() || (value - reference).abs() > 5e-3 * (1.0 + reference.abs()) {
                return Err(format!("ConvNeXt VJP: {value} != {reference}").into());
            }
        }
        Ok(())
    }

    pub async fn run() -> Result<String> {
        let runtime = WgpuRuntime::request_headless("vision.convnext_vjp.browser").await?;
        let device = TensorDevice::new(runtime)?;
        let mut block = ConvNeXtBlock::new("vision.browser_vjp", 2, (2, 2), -1.0, 1e-6)?;
        let input = Tensor::from_vec(
            2,
            8,
            vec![
                0.1, -0.3, 0.2, 0.4, -0.1, 0.5, -0.4, 0.3, 0.2, 0.4, -0.2, 0.1, 0.6, -0.1, 0.3,
                -0.5,
            ],
        )?;
        let seed = Tensor::from_vec(
            2,
            8,
            vec![
                0.5, -0.2, 0.7, 0.1, -0.3, 0.4, 0.2, -0.6, 0.1, 0.8, -0.4, 0.3, 0.2, -0.5, 0.6, 0.4,
            ],
        )?;
        let resident_input = device.upload(&[2, 2, 2, 2], input.data())?;
        let resident_seed = device.upload(&[2, 2, 2, 2], seed.data())?;
        let gradients = block.vjp_resident(&resident_input, &resident_seed)?;
        let expected_input = {
            let _policy = push_backend_policy(BackendPolicy::from_device_caps(DeviceCaps::cpu()));
            block.backward(&input, &seed)?
        };
        close(
            &gradients.input_gradient().snapshot()?.read_async().await?,
            expected_input.data(),
        )?;
        let mut references = Vec::new();
        block.visit_parameters(&mut |parameter| {
            references.push(parameter.gradient().unwrap().data().to_vec());
            Ok(())
        })?;
        if gradients.parameter_gradients().len() != 8 || references.len() != 8 {
            return Err("ConvNeXt VJP parameter count".into());
        }
        for (gradient, expected) in gradients.parameter_gradients().iter().zip(&references) {
            close(&gradient.snapshot()?.read_async().await?, expected)?;
        }
        Ok("{\"schema\":\"spiraltorch.convnext_block_vjp.browser.v1\",\"status\":\"passed\",\"gradient_tensors\":9,\"intermediate_readbacks\":0}".into())
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_convnext_block_vjp_checks() -> Result<String, wasm_bindgen::JsValue> {
    browser::run()
        .await
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}
