#[cfg(target_arch = "wasm32")]
mod browser {
    use st_backend_wgpu::{
        resident_tensor::{ResidentTensor, TensorDevice},
        runtime::WgpuRuntime,
    };
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    async fn read(tensor: &ResidentTensor) -> Result<Vec<f32>> {
        Ok(tensor.snapshot()?.read_async().await?)
    }

    fn close(actual: &[f32], expected: &[f32]) -> Result<()> {
        if actual.len() != expected.len() {
            return Err("LayerNorm output length".into());
        }
        for (&a, &b) in actual.iter().zip(expected) {
            if !a.is_finite()
                || (f64::from(a) - f64::from(b)).abs() > 2e-5 * (1. + f64::from(b).abs())
            {
                return Err(format!("LayerNorm: {a} != {b}").into());
            }
        }
        Ok(())
    }

    pub async fn run() -> Result<String> {
        let runtime = WgpuRuntime::request_headless("layer_norm.browser").await?;
        let device = TensorDevice::new(runtime.clone())?;
        let tiny = f32::from_bits(1);
        let attenuated = (f64::from(1e-20f32) * 0.5 / f64::from(f32::MAX).sqrt()) as f32;
        let mut cases = 0;
        for (shape, x, gamma, seed, epsilon, y, dx, dg, db) in [
            (
                [1, 2],
                vec![0., 1e-20],
                vec![f32::MAX; 2],
                vec![1.; 2],
                f32::MAX,
                vec![-attenuated * f32::MAX, attenuated * f32::MAX],
                vec![0.; 2],
                vec![-attenuated, attenuated],
                vec![1.; 2],
            ),
            (
                [3, 1],
                vec![1., 2., 3.],
                vec![f32::MAX],
                vec![f32::MAX, f32::MAX, -f32::MAX],
                1e-5,
                vec![0.; 3],
                vec![0.; 3],
                vec![0.],
                vec![f32::MAX],
            ),
            (
                [1, 2],
                vec![0., tiny],
                vec![1.; 2],
                vec![1.; 2],
                0.,
                vec![-1., 1.],
                vec![0.; 2],
                vec![-1., 1.],
                vec![1.; 2],
            ),
            (
                [1, 2],
                vec![0., tiny],
                vec![1.; 2],
                vec![0.25, -0.75],
                0.,
                vec![-1., 1.],
                vec![0.; 2],
                vec![-0.25, -0.75],
                vec![0.25, -0.75],
            ),
            (
                [1, 2],
                vec![f32::MAX, -f32::MAX],
                vec![1.; 2],
                vec![1.; 2],
                0.,
                vec![1., -1.],
                vec![0.; 2],
                vec![1., -1.],
                vec![1.; 2],
            ),
            (
                [1, 2],
                vec![f32::MAX; 2],
                vec![1.; 2],
                vec![1.; 2],
                tiny,
                vec![0.; 2],
                vec![0.; 2],
                vec![0.; 2],
                vec![1.; 2],
            ),
            (
                [0, 3],
                vec![],
                vec![1.; 3],
                vec![],
                0.,
                vec![],
                vec![],
                vec![0.; 3],
                vec![0.; 3],
            ),
        ] {
            let input = device.upload(&shape, &x)?;
            let gamma = device.upload(&[shape[1]], &gamma)?;
            let beta = device.upload(&[shape[1]], &vec![0.; shape[1]])?;
            let seed = device.upload(&shape, &seed)?;
            let tape = input.layer_norm_affine(&gamma, &beta, epsilon)?;
            close(&read(tape.value()).await?, &y)?;
            let expected = [dx, dg, db];
            for mask in 1..8 {
                let requested = [mask & 1 != 0, mask & 2 != 0, mask & 4 != 0];
                let grads = tape.backward(&seed, 1., requested)?;
                for i in 0..3 {
                    if grads[i].is_some() != requested[i] {
                        return Err("VJP presence differs from requested mask".into());
                    }
                    if let Some(value) = &grads[i] {
                        close(&read(value).await?, &expected[i])?;
                    } else if requested[i] {
                        return Err("Missing requested VJP".into());
                    }
                }
            }
            cases += 1;
        }
        let input = device.upload(&[3, 3], &[0.4, -0.8, 1.2, -0.3, 0.9, -1.1, 0.7, 0.1, -0.2])?;
        let target = input.layer_norm_affine(
            &device.upload(&[3], &[1.7, 0.5, -0.8])?,
            &device.upload(&[3], &[0.2, -0.3, 0.6])?,
            1e-5,
        )?;
        let mut gamma = device.upload(&[3], &[1.; 3])?;
        let mut beta = device.upload(&[3], &[0.; 3])?;
        let rate = device.upload(&[], &[-0.1])?;
        let mut first = None;
        let mut last = None;
        for _ in 0..400 {
            let tape = input.layer_norm_affine(&gamma, &beta, 1e-5)?;
            let loss = tape.value().mean_squared_error(target.value())?;
            let [_, dg, db] = tape.backward(loss.prediction_gradient(), 1., [false, true, true])?;
            gamma = gamma.add(&dg.unwrap().mul(&rate)?)?;
            beta = beta.add(&db.unwrap().mul(&rate)?)?;
            if first.is_none() {
                first = Some(loss.clone());
            }
            last = Some(loss);
        }
        let first = read(first.as_ref().unwrap().value()).await?[0];
        let last = read(last.as_ref().unwrap().value()).await?[0];
        if !last.is_finite() || last >= first * 1e-4 {
            return Err(format!("training {first} -> {last}").into());
        }
        let invalid = input.layer_norm_affine(&gamma, &beta, f32::NAN);
        if invalid.is_ok() {
            return Err("Invalid epsilon accepted".into());
        }
        let constant = device.upload(&[2, 1], &[0.; 2])?;
        let huge = device.upload(&[1], &[f32::MAX])?;
        let zero = device.upload(&[1], &[0.])?;
        let tape = constant.layer_norm_affine(&huge, &zero, 1e-5)?;
        let seed = device.upload(&[2, 1], &[f32::MAX; 2])?;
        let gradients = tape.backward(&seed, 1., [true; 3])?;
        for tensor in gradients.iter().flatten() {
            if read(tensor).await.is_ok() {
                return Err("Requested overflow escaped whole-operation guard".into());
            }
        }
        let dx = tape.backward(&seed, 1., [true, false, false])?[0]
            .clone()
            .unwrap();
        close(&read(&dx).await?, &[0.; 2])?;
        let bad_seed = seed.add(&seed)?.mul(&device.upload(&[], &[0.])?)?;
        let inherited = tape.backward(&bad_seed, 0., [false, false, true])?[2]
            .clone()
            .unwrap();
        if read(&inherited).await.is_ok() {
            return Err("Inherited failure escaped guard".into());
        }
        Ok(serde_json::to_string(&serde_json::json!({
            "schema": "spiraltorch.resident_layer_norm.browser.v1", "status": "passed",
            "adapter": format!("{:?}", runtime.adapter_info()), "cases": cases, "masks_per_case": 7,
            "training_steps": 400, "first_loss": first, "last_loss": last,
            "intermediate_readbacks": 0, "guard_checks": 4,
        }))?)
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_layer_norm_checks() -> Result<String, wasm_bindgen::JsValue> {
    browser::run()
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}
