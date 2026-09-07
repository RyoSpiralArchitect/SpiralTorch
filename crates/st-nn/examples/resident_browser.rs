//! Browser fixture using the real Rust Sequential lowering, not JS NN semantics.
#[cfg(target_arch = "wasm32")]
mod browser {
    use serde_json::json;
    use st_backend_wgpu::{resident_dense::DenseError, runtime::WgpuRuntime};
    use st_nn::{layers::Gelu, module::Module, resident::InferencePlan, Linear, Sequential};
    use st_tensor::{NdLayout, Tensor};
    use wasm_bindgen::prelude::*;

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn cpu_forward(model: &impl Module, input: &Tensor) -> st_tensor::PureResult<Tensor> {
        let _cpu = st_nn::push_backend_policy(st_nn::BackendPolicy::from_device_caps(
            st_core::backend::device_caps::DeviceCaps::cpu(),
        ));
        model.forward(input)
    }

    fn check(actual: &[f32], expected: &[f32]) -> Result<f32> {
        if actual.len() != expected.len() {
            return Err("output length differs".into());
        }
        let mut maximum = 0f32;
        for (&a, &b) in actual.iter().zip(expected) {
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-5 + 1e-4 * b.abs() {
                return Err(format!("output differs: {a} versus {b}").into());
            }
            maximum = maximum.max((a - b).abs());
        }
        Ok(maximum)
    }

    async fn run() -> Result<serde_json::Value> {
        let runtime = WgpuRuntime::request_headless("nn.browser.fixture").await?;
        let mut cases = Vec::new();
        for seed in [17, 29, 43] {
            for depth in [2, 16] {
                let mut model = Sequential::new();
                for i in 0..depth {
                    model.push(Linear::new(format!("linear_{i}"), 7, 7)?);
                    model.push(Gelu::new());
                }
                let mut counter = seed;
                model.visit_parameters_mut(&mut |parameter| {
                    for value in parameter.value_mut().data_mut() {
                        *value = ((counter * 17 % 23) as f32 - 11.0) / 64.0;
                        counter += 1;
                    }
                    Ok(())
                })?;
                let shape = [2, 3, 7];
                let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&shape)?)?;
                let mut gpu = plan.compile_wgpu(runtime.clone())?;
                if !matches!(gpu.dispatch(), Err(DenseError::MissingInput))
                    || !matches!(gpu.snapshot(), Err(DenseError::StaleOutput))
                {
                    return Err("missing input/output was accepted".into());
                }
                let input = Tensor::from_fn(6, 7, |r, c| (r as f32 - c as f32) / 8.0)?;
                let expected = cpu_forward(&model, &input)?;
                gpu.upload(input.data())?;
                gpu.dispatch()?;
                let first = gpu.snapshot()?;
                let generation = gpu.generation();
                if gpu.upload(&[f32::NAN; 42]).is_ok()
                    || gpu.upload(&[0.0; 41]).is_ok()
                    || gpu.generation() != generation
                {
                    return Err("invalid upload changed resident input".into());
                }
                let unchanged = gpu.snapshot()?;
                let next = Tensor::from_fn(6, 7, |r, c| (r + c) as f32 / 16.0)?;
                let expected_next = cpu_forward(&model, &next)?;
                gpu.upload(next.data())?;
                if !matches!(gpu.snapshot(), Err(DenseError::StaleOutput)) {
                    return Err("stale output was accepted".into());
                }
                gpu.dispatch()?;
                let second = gpu.snapshot()?;
                drop(gpu);
                if first.layout().shape() != shape || first.generation() != generation {
                    return Err("snapshot metadata changed".into());
                }
                let error = check(&first.read_async().await?, expected.data())?
                    .max(check(&unchanged.read_async().await?, expected.data())?)
                    .max(check(&second.read_async().await?, expected_next.data())?);
                cases.push(json!({"seed":seed,"shape":shape,"source_operations":depth*2,
                    "gpu_stages":depth,"max_abs_error":error,"snapshots":"owned across reupload and drop"}));
            }
        }
        let mut model = Sequential::new();
        model.push(Linear::new("guard", 1, 1)?);
        model.push(Gelu::new());
        model.visit_parameters_mut(&mut |parameter| {
            let value = if parameter.name().ends_with("weight") {
                1.0
            } else {
                0.0
            };
            parameter.value_mut().data_mut().fill(value);
            Ok(())
        })?;
        let plan = InferencePlan::from_module(&model, NdLayout::contiguous(&[1])?)?;
        let mut gpu = plan.compile_wgpu(runtime.clone())?;
        for value in [1e20f32, -1e20, 1e13, -1e13] {
            if cpu_forward(&model, &Tensor::from_vec(1, 1, vec![value])?).is_ok() {
                return Err("CPU GELU accepted overflowing intermediate".into());
            }
            gpu.upload(&[value])?;
            gpu.dispatch()?;
            let failure = gpu.snapshot()?;
            gpu.upload(&[0.25])?;
            gpu.dispatch()?;
            let success = gpu.snapshot()?;
            if !matches!(
                failure.read_async().await,
                Err(DenseError::NonFiniteIntermediate { stage: 0, .. })
            ) {
                return Err("GPU GELU hid an overflowing intermediate".into());
            }
            check(
                &success.read_async().await?,
                cpu_forward(&model, &Tensor::from_vec(1, 1, vec![0.25])?)?.data(),
            )?;
        }
        Ok(
            json!({"status":"passed","cases":cases,"intermediate_guard_cases":4,
            "adapter":{"name":runtime.adapter_info().name,"backend":format!("{:?}", runtime.adapter_info().backend)},
            "build_manifest":serde_json::from_str::<serde_json::Value>(st_core::build_manifest_json())?,
            "boundary":"Rust Sequential -> InferencePlan -> shared ResidentDense, WASM CPU oracle; correctness only, not browser timing or training"}),
        )
    }

    #[wasm_bindgen]
    pub async fn run_resident_nn_fixture() -> std::result::Result<String, JsValue> {
        run()
            .await
            .map(|value| value.to_string())
            .map_err(|error| JsValue::from_str(&error.to_string()))
    }
}
