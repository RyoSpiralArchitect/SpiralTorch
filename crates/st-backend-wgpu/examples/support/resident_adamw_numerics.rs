use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::TensorDevice, resident_training::parameters::ResidentParameters,
};
use st_kernel_contracts::adamw::{AdamWConfig, AdamWSlot, AdamWState, AdamWStep};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn comparison(actual: &[f32], expected: &Value) -> Value {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f64;
    let mut failures = 0;
    for (&actual, expected) in actual.iter().zip(expected) {
        let expected = expected.as_f64().unwrap();
        let error = (f64::from(actual) - expected).abs();
        max_abs = max_abs.max(error);
        if !actual.is_finite() || error > 3e-6 + 5e-5 * expected.abs() {
            failures += 1;
        }
    }
    json!({"max_abs": max_abs, "failures": failures})
}

pub async fn run(device: &TensorDevice) -> Result<Value> {
    let fixture: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/resident_adamw_numerics.json"
    )))?;
    let mut reports = Vec::new();
    let mut passed = true;
    for case in fixture["cases"].as_array().unwrap() {
        let config = floats(&case["config"]);
        let config = AdamWConfig::new(config[0], config[1], config[2])?;
        let step = AdamWStep::new(
            case["rate"].as_f64().unwrap() as f32,
            case["decay"].as_f64().unwrap() as f32,
        )?;
        let mut cpu = floats(&case["initial"]);
        let mut cpu_first = vec![0.0; cpu.len()];
        let mut cpu_second = vec![0.0; cpu.len()];
        let mut owner = ResidentParameters::new(vec![device.upload(&[cpu.len()], &cpu)?])?;
        owner.enable_adamw(config)?;
        let gradients: Vec<Vec<f32>> = case["gradient_period"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        let uploaded = gradients
            .iter()
            .map(|g| device.upload(&[g.len()], g))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut checkpoints = case["trajectory"].as_array().unwrap().iter().peekable();
        let mut trajectory = Vec::new();
        for cursor in 0..fixture["updates"].as_u64().unwrap() as u32 {
            let index = cursor as usize % gradients.len();
            for i in 0..cpu.len() {
                let candidate = step.candidate(
                    config,
                    cpu[i],
                    gradients[index][i],
                    cpu_first[i],
                    cpu_second[i],
                    cursor,
                )?;
                cpu[i] = candidate.value;
                cpu_first[i] = candidate.first;
                cpu_second[i] = candidate.second;
            }
            let gradient = owner
                .snapshot()
                .bind_gradients(vec![uploaded[index].clone()])?;
            let receipt = owner.adamw(&gradient, step)?.snapshot()?;
            // Every accepted update is checked, not inferred from submission.
            #[cfg(not(target_arch = "wasm32"))]
            let accepted = receipt.read()?;
            #[cfg(target_arch = "wasm32")]
            let accepted = receipt.read_async().await?;
            assert_eq!(accepted, u64::from(cursor + 1));
            if checkpoints.peek().unwrap()["steps"].as_u64() != Some(u64::from(cursor + 1)) {
                continue;
            }
            let expected = checkpoints.next().unwrap();
            let snapshot = owner.snapshot().values()[0].snapshot()?;
            #[cfg(not(target_arch = "wasm32"))]
            let values = snapshot.read()?;
            #[cfg(target_arch = "wasm32")]
            let values = snapshot.read_async().await?;
            let history = owner.adamw_snapshot()?.unwrap();
            #[cfg(not(target_arch = "wasm32"))]
            let history = history.read()?;
            #[cfg(target_arch = "wasm32")]
            let history = history.read_async().await?;
            let slot = &history.slots[0];
            assert_eq!(slot.steps, cursor + 1);
            let mut comparisons = serde_json::Map::new();
            for (engine, fields) in [
                ("gpu", [&values, &slot.first, &slot.second]),
                ("rust_cpu", [&cpu, &cpu_first, &cpu_second]),
            ] {
                for oracle in ["torch", "f64"] {
                    for (field, actual) in ["values", "first", "second"].into_iter().zip(fields) {
                        let result = comparison(actual, &expected[oracle][field]);
                        // f64 is an accuracy diagnostic; the unchanged Torch gate is mandatory.
                        if oracle == "torch" && result["failures"] != 0 {
                            passed = false;
                        }
                        comparisons.insert(format!("{engine}_{oracle}_{field}"), result);
                    }
                }
            }
            for (field, actual, expected) in [
                ("values", &values, &cpu),
                ("first", &slot.first, &cpu_first),
                ("second", &slot.second, &cpu_second),
            ] {
                let result = comparison(actual, &json!(expected));
                if result["failures"] != 0 {
                    passed = false;
                }
                comparisons.insert(format!("gpu_rust_cpu_{field}"), result);
            }
            trajectory.push(json!({"steps": cursor + 1, "comparisons": comparisons}));
        }
        assert!(checkpoints.next().is_none());
        reports.push(json!({"name": case["name"], "trajectory": trajectory}));
    }
    let probes = restored_clock_probes(device).await?;
    passed &= probes.iter().all(|p| p["passed"] == true);
    Ok(json!({"status": if passed {"passed"} else {"failed"},
        "schema": "spiraltorch.resident_adamw.numerics_report.v1",
        "torch": fixture["torch"], "updates_per_case": fixture["updates"],
        "required_oracles": ["torch_float32", "rust_cpu_float32"],
        "diagnostic_oracle": "scalar_f64",
        "atol": 3e-6, "rtol": 5e-5, "cases": reports,
        "restored_clock_probes": probes, "restored_clock_max_ulps": 8}))
}

async fn restored_clock_probes(device: &TensorDevice) -> Result<Vec<Value>> {
    let mut probes = Vec::new();
    for beta in [0.999, f32::from_bits(0x3f7fffff)] {
        for steps in [32u32, 1024, 1 << 24, u32::MAX] {
            let mut owner = ResidentParameters::from_restored_values(
                vec![device.upload(&[1], &[0.0])?],
                u64::from(steps - 1),
            )?;
            // Isolate the first-moment correction: zero gradient/second moment,
            // unit rate/epsilon, no decay. This is a valid restored synthetic state.
            owner.restore_adamw(&AdamWState {
                config: AdamWConfig::new(beta, 0.0, 1.0)?,
                slots: vec![AdamWSlot {
                    shape: vec![1],
                    first: vec![0.25],
                    second: vec![0.0],
                    steps: steps - 1,
                }],
            })?;
            let gradient = owner
                .snapshot()
                .bind_gradients(vec![device.upload(&[1], &[0.0])?])?;
            let receipt = owner
                .adamw(&gradient, AdamWStep::new(1.0, 0.0)?)?
                .snapshot()?;
            #[cfg(not(target_arch = "wasm32"))]
            let accepted = receipt.read()?;
            #[cfg(target_arch = "wasm32")]
            let accepted = receipt.read_async().await?;
            assert_eq!(accepted, u64::from(steps));
            let snapshot = owner.snapshot().values()[0].snapshot()?;
            #[cfg(not(target_arch = "wasm32"))]
            let actual = snapshot.read()?[0];
            #[cfg(target_arch = "wasm32")]
            let actual = snapshot.read_async().await?[0];
            let state = owner.adamw_snapshot()?.unwrap();
            #[cfg(not(target_arch = "wasm32"))]
            let state = state.read()?;
            #[cfg(target_arch = "wasm32")]
            let state = state.read_async().await?;
            assert_eq!(state.slots[0].steps, steps);
            let correction = -(f64::from(steps) * f64::from(beta).ln()).exp_m1();
            let expected = (-0.25 * f64::from(beta) / correction) as f32;
            let ulps = actual.to_bits().abs_diff(expected.to_bits());
            probes.push(json!({"beta_bits": beta.to_bits(), "steps": steps,
                "actual": actual, "expected": expected, "ulps": ulps,
                "passed": actual.is_finite() && ulps <= 8}));
        }
    }
    Ok(probes)
}
