//! Shared native/browser probe. The oracle is frozen PyTorch, not this kernel.
use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::{attention::AttentionMask, ResidentTensor, TensorDevice, TensorError},
    runtime::WgpuRuntime,
};
use st_kernel_contracts::{attention::AttentionSpec, layout::NdLayout};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn shape(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

fn spec(case: &Value) -> Result<AttentionSpec> {
    let qs = shape(&case["query_shape"]);
    let ks = shape(&case["key_shape"]);
    Ok(AttentionSpec::new(
        &qs,
        &ks,
        &ks,
        case["scale"].as_f64().unwrap() as f32,
        case["query_offset"]
            .as_u64()
            .map_or(AttentionMask::None, |v| AttentionMask::Causal {
                query_offset: v as usize,
            }),
    )?)
}

fn upload(
    device: &TensorDevice,
    shape: &[usize],
    data: &[f32],
    strided: bool,
) -> Result<ResidentTensor> {
    if !strided {
        return Ok(device.upload(shape, data)?);
    }
    let axes: Vec<_> = (0..shape.len()).rev().collect();
    let mut padded: Vec<_> = shape.iter().rev().copied().collect();
    padded[0] += 2;
    let storage = NdLayout::contiguous(&padded)?;
    let view = storage
        .narrow(0, 1, shape[shape.len() - 1])?
        .permute(&axes)?;
    let mut values = vec![0.; storage.len()];
    for (i, &value) in data.iter().enumerate() {
        values[view.storage_index(i).ok_or("fixture length")?] = value;
    }
    Ok(device
        .upload(&padded, &values)?
        .narrow(0, 1, shape[shape.len() - 1])?
        .permute(&axes)?)
}

async fn read(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = tensor.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let result = snapshot.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    let result = snapshot.read()?;
    Ok(result)
}

fn compare(label: &str, actual: &[f32], expected: &[f32]) -> Result<f64> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: length mismatch").into());
    }
    let mut maximum = 0f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let error = (f64::from(a) - f64::from(e)).abs();
        if !a.is_finite() || !e.is_finite() || error > 3e-6 + 5e-5 * f64::from(e).abs() {
            return Err(format!("{label}[{i}]: {a} != {e}").into());
        }
        maximum = maximum.max(error);
    }
    Ok(maximum)
}

async fn edge_checks(device: &TensorDevice) -> Result<Value> {
    for count in [127, 128, 131] {
        let d = 24;
        let q = device.upload(&[1, 1, 1, d], &vec![1.; d])?;
        let mut keys = vec![0.; count * d];
        keys[0] = 16_777_216.;
        keys[8] = 1.;
        keys[16] = -16_777_216.;
        let mut values = vec![0.; count * d];
        values[0] = 1.;
        let mut seed = vec![0.; d];
        seed[0] = 1.;
        let k = device.upload(&[1, 1, count, d], &keys)?;
        let v = device.upload(&[1, 1, count, d], &values)?;
        let upstream = device.upload(&[1, 1, 1, d], &seed)?;
        let output = q.scaled_dot_attention(&k, &v, 1., AttentionMask::None, None, None)?;
        let gradient =
            q.scaled_dot_attention_vjp(&k, &v, &upstream, 1., AttentionMask::None, None, None)?;
        let probability = read(&output).await?[0];
        let expected = if count >= 128 {
            1. / count as f32
        } else {
            1f32.exp() / (count as f32 - 1. + 1f32.exp())
        };
        compare("forward reduction control", &[probability], &[expected])?;
        compare(
            "backward must use forward scores",
            &[read(&gradient.value).await?[0]],
            &[probability],
        )?;
    }
    let good = device.upload(&[1], &[0.])?;
    let failed = device
        .upload(&[2], &[0., f32::MAX])?
        .mul(&device.upload(&[2], &[1., 2.])?)?
        .narrow(0, 0, 1)?;
    let mut guards = 0;
    for queries in [0, 1] {
        for slot in 0..6 {
            let input = |which, shape: &[usize]| {
                (if slot == which { &failed } else { &good }).broadcast_to(shape)
            };
            let result = input(0, &[2, 2, queries, 1])?.scaled_dot_attention_vjp(
                &input(1, &[2, 2, 3, 1])?,
                &input(2, &[2, 2, 3, 1])?,
                &input(5, &[2, 2, queries, 1])?,
                1.,
                AttentionMask::Causal { query_offset: 0 },
                Some(&input(3, &[2, 2, 3])?),
                Some(&input(4, &[2, 2, queries, 3])?),
            )?;
            for tensor in [
                result.query,
                result.key,
                result.value,
                result.z_bias.unwrap(),
                result.pair_bias.unwrap(),
            ] {
                if !matches!(read(&tensor).await, Err(e) if matches!(e.downcast_ref::<TensorError>(), Some(TensorError::NonFinite)))
                {
                    return Err("lost inherited attention guard".into());
                }
            }
            guards += 1;
        }
    }
    let qs = [1, 1, 1, 1];
    let ks = [1, 1, 2, 1];
    let upstream = device.upload(&qs, &[1e38])?;
    let mut ranges = Vec::new();
    for (name, q, k, v, scale, pair) in [
        (
            "wide_intermediates",
            1e-38,
            [-1e-38, 1e-38],
            [-1e38, 1e38],
            0.5,
            None,
        ),
        (
            "probability_underflow",
            0.,
            [1., 0.],
            [1e38, 0.],
            1.,
            Some([-120., 0.]),
        ),
    ] {
        let spec = AttentionSpec::new(&qs, &ks, &ks, scale, AttentionMask::None)?;
        let expected = st_kernel_contracts::attention::attention_vjp_reference(
            spec,
            &[q],
            &k,
            &v,
            None,
            pair.as_ref().map(|v| v.as_slice()),
            &[1e38],
        )?;
        let pg = pair
            .as_ref()
            .map(|v| device.upload(&[1, 1, 1, 2], v))
            .transpose()?;
        let actual = device.upload(&qs, &[q])?.scaled_dot_attention_vjp(
            &device.upload(&ks, &k)?,
            &device.upload(&ks, &v)?,
            &upstream,
            scale,
            AttentionMask::None,
            None,
            pg.as_ref(),
        )?;
        let mut errors = serde_json::Map::new();
        for (field, tensor, expected) in [
            ("query", actual.query, expected.query),
            ("key", actual.key, expected.key),
            ("value", actual.value, expected.value),
        ] {
            errors.insert(
                field.into(),
                json!(compare(field, &read(&tensor).await?, &expected)?),
            );
        }
        if let (Some(tensor), Some(expected)) = (actual.pair_bias, expected.pair_bias) {
            errors.insert(
                "pair_bias".into(),
                json!(compare("pair_bias", &read(&tensor).await?, &expected)?),
            );
        }
        ranges.push(
            json!({"name": name, "reference": "shared Rust f64 VJP", "max_abs_error": errors}),
        );
    }
    Ok(
        json!({"inherited_guard_cases": guards, "range_cases": ranges, "forward_reduction_cases": 3}),
    )
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("CPU adapter is not GPU evidence".into());
    }
    let device = TensorDevice::new(runtime.clone())?;
    let edges = edge_checks(&device).await?;
    let fixture: Value = serde_json::from_str(include_str!(
        "../fixtures/resident_attention_vjp_torch.json"
    ))?;
    if fixture["schema"] != "spiraltorch.resident_attention_vjp_torch.v1"
        || fixture["cases"].as_array().map(Vec::len) != Some(18)
    {
        return Err("unexpected fixture".into());
    }
    let mut checks = Vec::new();
    for case in fixture["cases"].as_array().unwrap() {
        let spec = spec(case)?;
        for strided in [false, true] {
            let q = upload(
                &device,
                &spec.query_shape(),
                &floats(&case["query"]),
                strided,
            )?;
            let k = upload(&device, &spec.key_shape(), &floats(&case["key"]), strided)?;
            let v = upload(&device, &spec.key_shape(), &floats(&case["value"]), strided)?;
            let u = upload(
                &device,
                &spec.query_shape(),
                &floats(&case["upstream"]),
                strided,
            )?;
            let z = if case["z_bias"].is_null() {
                None
            } else {
                Some(upload(
                    &device,
                    &spec.z_bias_shape(),
                    &floats(&case["z_bias"]),
                    strided,
                )?)
            };
            let pair = if case["pair_bias"].is_null() {
                None
            } else {
                Some(upload(
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
            let output = q.scaled_dot_attention(
                &k,
                &v,
                spec.scale(),
                spec.mask(),
                z.as_ref(),
                pair.as_ref(),
            )?;
            let gradients = q.scaled_dot_attention_vjp(
                &k,
                &v,
                &u,
                spec.scale(),
                spec.mask(),
                z.as_ref(),
                pair.as_ref(),
            )?;
            if let Some(error) = runtime.context().device().pop_error_scope().await {
                return Err(format!("WebGPU validation: {error}").into());
            }
            drop((q, k, v, u, z, pair));
            let mut errors = serde_json::Map::new();
            errors.insert(
                "forward".into(),
                json!(compare(
                    "forward",
                    &read(&output).await?,
                    &floats(&case["expected"])
                )?),
            );
            for (name, tensor, shape) in [
                ("query", Some(gradients.query), spec.query_shape().to_vec()),
                ("key", Some(gradients.key), spec.key_shape().to_vec()),
                ("value", Some(gradients.value), spec.key_shape().to_vec()),
                ("z_bias", gradients.z_bias, spec.z_bias_shape().to_vec()),
                (
                    "pair_bias",
                    gradients.pair_bias,
                    spec.pair_bias_shape().to_vec(),
                ),
            ] {
                if tensor.is_none() != case["gradients"][name].is_null() {
                    return Err("bias presence mismatch".into());
                }
                if let Some(tensor) = tensor {
                    if tensor.layout().shape() != shape {
                        return Err("gradient shape mismatch".into());
                    }
                    errors.insert(
                        name.into(),
                        json!(compare(
                            name,
                            &read(&tensor).await?,
                            &floats(&case["gradients"][name])
                        )?),
                    );
                }
            }
            checks.push(json!({"name": case["name"], "strided": strided, "max_abs_error": errors}));
        }
    }
    let learning = &fixture["learning"];
    let spec = spec(learning)?;
    let names = ["query", "key", "value", "z_bias", "pair_bias"];
    let shapes = [
        spec.query_shape().to_vec(),
        spec.key_shape().to_vec(),
        spec.key_shape().to_vec(),
        spec.z_bias_shape().to_vec(),
        spec.pair_bias_shape().to_vec(),
    ];
    let mut parameters = names
        .iter()
        .zip(&shapes)
        .map(|(name, shape)| upload(&device, shape, &floats(&learning[name]), false))
        .collect::<Result<Vec<_>>>()?;
    let negative_target = device.upload(
        &spec.query_shape(),
        &floats(&learning["target"])
            .iter()
            .map(|v| -v)
            .collect::<Vec<_>>(),
    )?;
    let mse_scale = device.upload(&[], &[2. / parameters[0].layout().len() as f32])?;
    let negative_lr = device.upload(
        &[],
        &[-(learning["learning_rate"].as_f64().unwrap() as f32)],
    )?;
    let steps = learning["steps"].as_u64().unwrap() as usize;
    if steps != 16 || learning["trace"].as_array().map(Vec::len) != Some(steps) {
        return Err("incomplete learning oracle".into());
    }
    let mut outputs = Vec::new();
    runtime
        .context()
        .device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    for _ in 0..steps {
        let [q, k, v, z, pair] = parameters.as_slice() else {
            unreachable!()
        };
        let output =
            q.scaled_dot_attention(k, v, spec.scale(), spec.mask(), Some(z), Some(pair))?;
        let upstream = output.add(&negative_target)?.mul(&mse_scale)?;
        let g = q.scaled_dot_attention_vjp(
            k,
            v,
            &upstream,
            spec.scale(),
            spec.mask(),
            Some(z),
            Some(pair),
        )?;
        parameters = parameters
            .iter()
            .zip([
                g.query,
                g.key,
                g.value,
                g.z_bias.unwrap(),
                g.pair_bias.unwrap(),
            ])
            .map(|(p, g)| Ok(p.add(&g.mul(&negative_lr)?)?))
            .collect::<Result<Vec<_>>>()?;
        outputs.push(output);
    }
    if let Some(error) = runtime.context().device().pop_error_scope().await {
        return Err(format!("learning validation: {error}").into());
    }
    // All 16 updates above are GPU-only. Host reads below are validation only.
    let mut trace = Vec::new();
    let target = floats(&learning["target"]);
    for (step, output) in outputs.iter().enumerate() {
        let actual = read(output).await?;
        let error = compare(
            "learning output",
            &actual,
            &floats(&learning["trace"][step]["output"]),
        )?;
        let loss = actual
            .iter()
            .zip(&target)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / target.len() as f64;
        if (loss - learning["trace"][step]["loss"].as_f64().unwrap()).abs() > 3e-6 {
            return Err("learning loss mismatch".into());
        }
        trace.push(json!({"step": step, "loss": loss, "max_abs_error": error}));
    }
    let mut final_errors = serde_json::Map::new();
    for (name, tensor) in names.iter().zip(&parameters) {
        final_errors.insert(
            (*name).into(),
            json!(compare(
                name,
                &read(tensor).await?,
                &floats(&learning["final"][name])
            )?),
        );
    }
    Ok(
        json!({"schema": "spiraltorch.resident_attention_vjp.validation.v1", "passed": true,
        "adapter": format!("{:?}", runtime.adapter_info()), "torch_version": fixture["torch_version"],
        "checks": checks, "edge_checks": edges, "learning": {"steps": steps, "trace": trace, "final_parameter_errors": final_errors,
        "validation_readbacks": "after all updates"},
        "scope": "synthetic gradient/update parity; no speed, full-decoder or quality claim"}),
    )
}
