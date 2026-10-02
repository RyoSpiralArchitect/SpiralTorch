use st_backend_wgpu::{resident_tensor::TensorDevice, runtime::WgpuRuntime};
use st_kernel_contracts::attention::AttentionMask;
use st_nn::{
    resident::AttentionInferencePlan,
    z_rba::attention::{SimpleZFrame, ZIndex, ZMetricWeights, ZRBFAttention},
    Tensor,
};
use st_tensor::NdLayout;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn floats(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn close(actual: &[f32], expected: &[f32], name: &str) -> Result<f32> {
    if actual.len() != expected.len() {
        return Err(format!("{name}: output length").into());
    }
    let mut error = 0f32;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || !b.is_finite() || (a - b).abs() > 3e-6 + 3e-5 * b.abs() {
            return Err(format!("{name}[{index}]: {a} != {b}").into());
        }
        error = error.max((a - b).abs());
    }
    Ok(error)
}

/// Both clients execute this Rust orchestration and the same frozen oracle.
/// Geometry is prepared once on the host; no activation is read back until
/// after the output projection. No timing or quality claim is made by this probe.
pub async fn run() -> Result<serde_json::Value> {
    run_fixture(serde_json::from_str(include_str!(
        "../../tests/fixtures/attention_chain_torch.json"
    ))?)
    .await
}

/// Test-only supplied fixture lets the browser exercise larger kernel regimes
/// without embedding benchmark arrays into the library or duplicating the math.
pub async fn run_fixture(fixture: serde_json::Value) -> Result<serde_json::Value> {
    if fixture["schema"] != "spiraltorch.attention_chain_torch.v1" {
        return Err("wrong fixture schema".into());
    }
    let scenarios = fixture["scenarios"].as_array().ok_or("missing scenarios")?;
    let expected_checks = scenarios
        .iter()
        .map(|s| s["cases"].as_array().map_or(0, Vec::len))
        .sum::<usize>();
    if scenarios.is_empty() || expected_checks == 0 {
        return Err("empty fixture".into());
    }
    let runtime = WgpuRuntime::request_headless("nn.attention.chain.fixture").await?;
    #[cfg(not(target_arch = "wasm32"))]
    if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
        return Err("a real GPU is required for the native fixture".into());
    }
    let device = TensorDevice::new(runtime.clone())?;
    let mut checks = Vec::new();
    let mut geometry_checks = Vec::new();
    for scenario in fixture["scenarios"].as_array().unwrap() {
        let name = scenario["name"].as_str().unwrap();
        let shape: Vec<_> = scenario["input_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let (batch, sequence, inner) = (shape[0], shape[1], shape[2]);
        let heads = scenario["heads"].as_u64().unwrap() as usize;
        let width = scenario["width"].as_u64().unwrap() as usize;
        let out = scenario["output_width"].as_u64().unwrap() as usize;
        let weights: Vec<_> = (0..4)
            .map(|i| {
                Tensor::from_vec(
                    if i < 3 { inner } else { width },
                    if i < 3 { width } else { out },
                    floats(&scenario["weights"][i]),
                )
            })
            .collect::<std::result::Result<_, _>>()?;
        let biases: Vec<_> = (0..4)
            .map(|i| {
                Tensor::from_vec(
                    1,
                    if i < 3 { width } else { out },
                    floats(&scenario["biases"][i]),
                )
            })
            .collect::<std::result::Result<_, _>>()?;
        let indices: Vec<_> = scenario["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| ZIndex {
                band: v[0].as_u64().unwrap() as usize,
                sheet: v[1].as_u64().unwrap() as usize,
                echo: v[2].as_u64().unwrap() as usize,
            })
            .collect();
        let dims: Vec<_> = scenario["frame_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let frame = SimpleZFrame::new(dims[0], dims[1], dims[2]);
        let geometry = ZRBFAttention::new(width, heads, ZMetricWeights::default(), true)?;
        let kernel = geometry.kernel_bias(&frame, &indices, &indices)?;
        geometry_checks.push(serde_json::json!({"name": name, "max_abs_error": close(kernel.data(), &floats(&scenario["expected_kernel"]), name)?}));
        let pair = device.upload(&[1, heads, sequence, sequence], kernel.data())?;
        let input = device.upload(&shape, &floats(&scenario["input"]))?;
        for causal in [false, true] {
            let mask = if causal {
                AttentionMask::Causal { query_offset: 0 }
            } else {
                AttentionMask::None
            };
            let plan = AttentionInferencePlan::from_parameters(
                NdLayout::contiguous(&shape)?,
                heads,
                mask,
                std::array::from_fn(|i| (&weights[i], &biases[i])),
            )?;
            let mut compiled = plan.compile_wgpu(runtime.clone())?;
            for case in scenario["cases"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c["causal"].as_bool().unwrap() == causal)
            {
                let pair = if let Some(strength) = case["geometry_strength"].as_f64() {
                    Some(
                        pair.mul(&device.upload(&[1], &[strength as f32])?)?
                            .broadcast_to(&[batch, heads, sequence, sequence])?,
                    )
                } else {
                    None
                };
                let output = compiled.forward(&input, None, pair.as_ref())?;
                #[cfg(not(target_arch = "wasm32"))]
                let actual = output.snapshot()?.read()?;
                #[cfg(target_arch = "wasm32")]
                let actual = output.snapshot()?.read_async().await?;
                let case_name = format!("{name}/{}", case["name"].as_str().unwrap());
                let error = close(&actual, &floats(&case["expected"]), &case_name)?;
                checks.push(serde_json::json!({"name": case_name, "max_abs_error": error}));
            }
        }
    }
    if checks.len() != expected_checks || geometry_checks.len() != scenarios.len() {
        return Err("incomplete attention-chain fixture".into());
    }
    Ok(serde_json::json!({
        "schema": "spiraltorch.attention_chain.v1", "passed": true,
        "adapter": format!("{:?}", runtime.adapter_info()),
        "reference_torch_version": fixture["torch_version"],
        "checks": checks, "geometry_checks": geometry_checks,
        "scope": "full projection/attention forward and independent geometry parity; no speed or quality claim",
    }))
}

#[cfg(test)]
mod tests {
    #[test]
    fn invalid_reference_values_never_pass_the_numerical_gate() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(super::close(&[0.], &[value], "invalid reference").is_err());
        }
    }
}
