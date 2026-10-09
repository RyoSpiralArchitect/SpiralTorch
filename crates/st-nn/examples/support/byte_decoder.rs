//! Complete byte-model oracle, shared by native WGPU and browser WASM.
use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice, TensorError},
    resident_training::TrainingError,
    runtime::WgpuRuntime,
};
use st_kernel_contracts::classification::{ClassReduction, CrossEntropySpec};
use st_nn::{
    resident::{
        AttentionInferencePlan, AttentionMask, ByteDecoderBias, ByteDecoderGeometryPlan,
        ByteDecoderPairMetric, ByteDecoderPlan, ByteLmBatch, InferenceError, InferenceOp,
        InferencePlan, ResidualAttentionPlan, ToposResonatorKernel,
    },
    Tensor,
};
use st_tensor::NdLayout;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type OwnedBias = (Option<ResidentTensor>, Option<ResidentTensor>);

#[path = "byte_decoder/checkpoint_controls.rs"]
mod checkpoint_controls;
#[path = "byte_decoder/geometry_controls.rs"]
mod geometry_controls;
#[path = "byte_decoder/parameter_rate_controls.rs"]
mod parameter_rate_controls;

pub async fn run_parameter_rates(runtime: WgpuRuntime) -> Result<Value> {
    parameter_rate_controls::run(runtime).await
}

fn data(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
fn sizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}
fn tensor(p: &Value) -> Result<Tensor> {
    let s = sizes(&p["shape"]);
    Ok(Tensor::from_vec(
        if s.len() == 1 { 1 } else { s[0] },
        *s.last().unwrap(),
        data(&p["values"]),
    )?)
}
fn norm(p: &[Value]) -> Result<InferenceOp> {
    Ok(InferenceOp::LayerNorm {
        gain: tensor(&p[0])?,
        bias: tensor(&p[1])?,
        epsilon: 1e-5,
    })
}

fn plan(case: &Value, steps: usize) -> Result<ByteDecoderPlan> {
    let c = &case["config"];
    let b = c["batch"].as_u64().unwrap() as usize;
    let width = c["width"].as_u64().unwrap() as usize;
    let hidden = c["hidden"].as_u64().unwrap() as usize;
    let heads = c["heads"].as_u64().unwrap() as usize;
    let layout = NdLayout::contiguous(&[b, steps, width])?;
    let parameters = case["parameters"].as_array().unwrap();
    let mut offset = 2;
    let geometry = if !c["causal_geometry"].is_null() {
        let projection = InferencePlan::from_operations(
            layout.clone(),
            vec![InferenceOp::Linear {
                weight: tensor(&parameters[2])?,
                bias: tensor(&parameters[3])?,
            }],
        )?;
        offset = 6 + c["blocks"].as_array().unwrap().len();
        let metric = match c["causal_geometry"].get("pair_metric") {
            None => ByteDecoderPairMetric::PoincareSquared,
            Some(Value::String(s)) if s == "poincare_squared.v1" => {
                ByteDecoderPairMetric::PoincareSquared
            }
            Some(Value::String(s)) if s == "euclidean_chord_squared.v1" => {
                ByteDecoderPairMetric::EuclideanChordSquared
            }
            _ => return Err("unknown geometry pair metric".into()),
        };
        Some(
            ByteDecoderGeometryPlan::new(
                &projection,
                &data(&parameters[4]["values"]),
                &data(&parameters[5]["values"]),
                &parameters[6..offset]
                    .iter()
                    .map(|p| data(&p["values"]))
                    .collect::<Vec<_>>(),
                c["causal_geometry"]["curvature"].as_f64().unwrap() as f32,
            )?
            .with_pair_metric(metric),
        )
    } else {
        None
    };
    let mut blocks = Vec::new();
    for block in c["blocks"].as_array().unwrap() {
        let topos = block["topos"].as_bool().unwrap();
        let count = if topos { 13 } else { 12 };
        let p = &parameters[offset..offset + count];
        let pre = InferencePlan::from_operations(layout.clone(), vec![norm(p)?])?;
        let fused = data(&p[2]["values"]);
        let fused_bias = data(&p[3]["values"]);
        let mut projections = Vec::new();
        for i in 0..3 {
            let weights: Vec<_> = (0..width)
                .flat_map(|row| {
                    fused[row * 3 * width + i * width..row * 3 * width + (i + 1) * width]
                        .iter()
                        .copied()
                })
                .collect();
            projections.push((
                Tensor::from_vec(width, width, weights)?,
                Tensor::from_vec(1, width, fused_bias[i * width..(i + 1) * width].to_vec())?,
            ));
        }
        projections.push((tensor(&p[4])?, tensor(&p[5])?));
        let attention = AttentionInferencePlan::from_parameters(
            layout.clone(),
            heads,
            AttentionMask::Causal { query_offset: 0 },
            std::array::from_fn(|i| (&projections[i].0, &projections[i].1)),
        )?;
        let mut operations = vec![
            norm(&p[6..])?,
            InferenceOp::Linear {
                weight: tensor(&p[8])?,
                bias: tensor(&p[9])?,
            },
            InferenceOp::Gelu,
        ];
        if topos {
            operations.push(InferenceOp::ToposResonator {
                gate: tensor(&p[10])?,
                kernel: ToposResonatorKernel::new(0.2, 0.12, 0.3, 4)?,
                max_volume: b * steps * hidden,
            });
        }
        operations.push(InferenceOp::Linear {
            weight: tensor(&p[count - 2])?,
            bias: tensor(&p[count - 1])?,
        });
        let feed = InferencePlan::from_operations(layout.clone(), operations)?;
        blocks.push(ResidualAttentionPlan::from_plans(&pre, &attention, &feed)?);
        offset += count;
    }
    let p = &parameters[offset..];
    if p.len() != 4 {
        return Err("incomplete decoder head".into());
    }
    let head = InferencePlan::from_operations(
        layout,
        vec![
            norm(p)?,
            InferenceOp::Linear {
                weight: tensor(&p[2])?,
                bias: tensor(&p[3])?,
            },
        ],
    )?;
    let plan = ByteDecoderPlan::from_plans(
        &tensor(&parameters[0])?,
        &tensor(&parameters[1])?,
        &blocks,
        &head,
    )?;
    Ok(match geometry {
        Some(g) => plan.with_causal_geometry(g)?,
        None => plan,
    })
}

fn batch(windows: &[Vec<u8>], steps: usize) -> Result<ByteLmBatch> {
    let docs: Vec<_> = windows.iter().map(Vec::as_slice).collect();
    Ok(ByteLmBatch::from_documents(
        &docs,
        &(0..docs.len()).map(|i| (i, 0)).collect::<Vec<_>>(),
        steps,
    )?)
}

fn biases(device: &TensorDevice, case: &Value, steps: usize) -> Result<Vec<OwnedBias>> {
    let c = &case["config"];
    let b = c["batch"].as_u64().unwrap() as usize;
    let h = c["heads"].as_u64().unwrap() as usize;
    let full = c["steps"].as_u64().unwrap() as usize;
    case["biases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|bias| {
            Ok((
                if bias["z"].is_null() {
                    None
                } else {
                    Some(
                        device
                            .upload(&[b, h, full], &data(&bias["z"]))?
                            .narrow(2, 0, steps)?,
                    )
                },
                if bias["pair"].is_null() {
                    None
                } else {
                    Some(
                        device
                            .upload(&[b, h, full, full], &data(&bias["pair"]))?
                            .narrow(2, 0, steps)?
                            .narrow(3, 0, steps)?,
                    )
                },
            ))
        })
        .collect()
}

fn borrowed(biases: &[OwnedBias]) -> Vec<ByteDecoderBias<'_>> {
    biases
        .iter()
        .map(|(z, pair)| ByteDecoderBias {
            z_bias: z.as_ref(),
            pair_bias: pair.as_ref(),
        })
        .collect()
}

async fn read(t: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = t.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    Ok(values)
}

async fn read_many(device: &TensorDevice, tensors: &[ResidentTensor]) -> Result<Vec<Vec<f32>>> {
    let snapshot = device.snapshot_many(&tensors.iter().collect::<Vec<_>>())?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    Ok(values)
}

fn close(actual: &[f32], expected: &[f32], label: &str) -> Result<f64> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: length mismatch").into());
    }
    let mut maximum = 0f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let delta = (f64::from(a) - f64::from(e)).abs();
        if !a.is_finite() || !e.is_finite() || delta > 3e-6 + 5e-5 * f64::from(e).abs() {
            return Err(format!("{label}[{i}]: {a} != {e}").into());
        }
        maximum = maximum.max(delta);
    }
    Ok(maximum)
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .map(|v| v.to_bits())
            .eq(b.iter().map(|v| v.to_bits()))
}

async fn causal_checks(runtime: &WgpuRuntime, case: &Value) -> Result<Value> {
    let steps = 4;
    let windows = vec![vec![1, 2, 3, 4, 5], vec![10, 11, 12, 13, 14]];
    let p = plan(case, steps)?;
    let mut model = p.compile_training_wgpu(runtime.clone())?;
    let geometry = biases(model.tensor_device(), case, steps)?;
    let original = model.prepare_batch(&batch(&windows, steps)?)?;
    let before = model.forward_with_external_biases(&original, &borrowed(&geometry))?;
    let baseline = read(before.prediction()).await?;
    let mut changed = windows.clone();
    changed[0][2..].copy_from_slice(&[203, 204, 205]);
    let suffix = model.prepare_batch(&batch(&changed, steps)?)?;
    let after = model.forward_with_external_biases(&suffix, &borrowed(&geometry))?;
    let after = read(after.prediction()).await?;
    let prefix_error = close(&after[..2 * 256], &baseline[..2 * 256], "causal prefix")?;
    let row_error = close(
        &after[steps * 256..],
        &baseline[steps * 256..],
        "document isolation",
    )?;
    if !after[2 * 256..4 * 256]
        .iter()
        .zip(&baseline[2 * 256..4 * 256])
        .any(|(a, b)| (a - b).abs() > 1e-6)
    {
        return Err("causality control is insensitive to changed input".into());
    }
    let mut short = plan(case, 2)?.compile_training_wgpu(runtime.clone())?;
    let short_batch = short.prepare_batch(&batch(&windows, 2)?)?;
    let short_bias = biases(short.tensor_device(), case, 2)?;
    let prefix = short.forward_with_external_biases(&short_batch, &borrowed(&short_bias))?;
    let prefix = read(prefix.prediction()).await?;
    let expected: Vec<_> = (0..2)
        .flat_map(|b| {
            baseline[b * steps * 256..(b * steps + 2) * 256]
                .iter()
                .copied()
        })
        .collect();
    let extension_error = close(&prefix, &expected, "prefix alone versus extension")?;
    let forward = model.forward_with_external_biases(&original, &borrowed(&geometry))?;
    let mut seed = vec![0.; 2 * steps * 256];
    for b in 0..2 {
        for t in 0..2 {
            for c in 0..256 {
                seed[(b * steps + t) * 256 + c] = (c % 17) as f32 * 0.001;
            }
        }
    }
    let seed = model
        .tensor_device()
        .upload(model.output_layout().shape(), &seed)?;
    let gradient = model.backward(&forward, &seed)?;
    let table = read(&gradient.parameter_gradients()[0]).await?;
    let width = p.embedding_width();
    for id in [3, 4, 12, 13] {
        if table[id * width..(id + 1) * width].iter().any(|&v| v != 0.) {
            return Err("future byte has a prefix-loss gradient".into());
        }
    }
    let embedded = read(gradient.embedding_output_gradient()).await?;
    for b in 0..2 {
        if embedded[(b * steps + 2) * width..(b + 1) * steps * width]
            .iter()
            .any(|&v| v != 0.)
        {
            return Err("future embedding output has a prefix cotangent".into());
        }
    }
    Ok(
        json!({"prefix_max_abs_error": prefix_error, "other_document_max_abs_error": row_error,
        "extension_max_abs_error": extension_error, "suffix_gradient_zero": true, "sensitivity_control": true}),
    )
}

async fn tape_checks(runtime: &WgpuRuntime, case: &Value) -> Result<Value> {
    let p = plan(case, 4)?;
    let mut a = p.compile_training_wgpu(runtime.clone())?;
    let mut b = p.compile_training_wgpu(runtime.clone())?;
    let host = ByteLmBatch::from_windows(&[b"abaca", b"xyxyz"])?;
    let batch = a.prepare_batch(&host)?;
    let a1 = a.forward(&batch)?;
    let b1 = b.forward(&batch)?;
    let good_seed = a
        .tensor_device()
        .upload(a.output_layout().shape(), &data(&case["cotangent"]))?;
    if !matches!(
        a.backward(&b1, &good_seed),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ) {
        return Err("foreign same-revision decoder tape accepted".into());
    }
    let frozen = read(a1.prediction()).await?;
    let changed = a.prepare_batch(&ByteLmBatch::from_windows(&[b"zzxqy", b"mnbvx"])?)?;
    let a2 = a.forward(&changed)?;
    let changed_output = read(a2.prediction()).await?;
    if !frozen
        .iter()
        .zip(&changed_output)
        .any(|(a, b)| (a - b).abs() > 1e-6)
    {
        return Err("retained-output control is insensitive to changed input".into());
    }
    if !matches!(
        a.backward(&a1, &good_seed),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ) {
        return Err("superseded decoder tape accepted".into());
    }
    if !same_bits(&frozen, &read(a1.prediction()).await?) {
        return Err("old decoder output changed".into());
    }
    let mut bad_bias = vec![ByteDecoderBias::default(); p.block_count()];
    let wrong = a.tensor_device().upload(&[1], &[0.])?;
    bad_bias.last_mut().unwrap().z_bias = Some(&wrong);
    if a.forward_with_external_biases(&batch, &bad_bias).is_ok() {
        return Err("bad final-block bias accepted".into());
    }
    let good = a.backward(&a2, &good_seed)?;
    let saved = read_many(a.tensor_device(), good.parameter_gradients()).await?;
    let huge = a.tensor_device().upload(
        a.output_layout().shape(),
        &vec![f32::MAX; a.output_layout().len()],
    )?;
    let bad_seed = huge.mul(&huge)?;
    let bad = a.backward(&a2, &bad_seed)?;
    if bad.parameter_gradients().len() != p.parameter_layout().len() {
        return Err("truncated failed decoder VJP".into());
    }
    for g in bad
        .parameter_gradients()
        .iter()
        .chain([bad.embedding_output_gradient()])
    {
        if !matches!(read(g).await, Err(e) if matches!(e.downcast_ref::<TensorError>(), Some(TensorError::NonFinite)))
        {
            return Err("failed decoder VJP is readable".into());
        }
    }
    let retained = read_many(a.tensor_device(), good.parameter_gradients()).await?;
    if saved.len() != retained.len() || saved.iter().zip(&retained).any(|(a, b)| !same_bits(a, b)) {
        return Err("invalid pullback changed retained gradients".into());
    }
    let repeated = a.backward(&a2, &good_seed)?;
    let repeated = read_many(a.tensor_device(), repeated.parameter_gradients()).await?;
    if saved.len() != repeated.len() || saved.iter().zip(&repeated).any(|(a, b)| !same_bits(a, b)) {
        return Err("bad pullback poisoned a later good pullback".into());
    }
    let changed_seed = a.tensor_device().upload(
        a.output_layout().shape(),
        &data(&case["cotangent"])
            .into_iter()
            .map(|v| -v)
            .collect::<Vec<_>>(),
    )?;
    let changed_gradient = a.backward(&a2, &changed_seed)?;
    let changed_gradient =
        read_many(a.tensor_device(), changed_gradient.parameter_gradients()).await?;
    if !saved
        .iter()
        .flatten()
        .zip(changed_gradient.iter().flatten())
        .any(|(a, b)| (a - b).abs() > 1e-6)
    {
        return Err("retained-gradient control is insensitive to changed cotangent".into());
    }
    let retained = read_many(a.tensor_device(), good.parameter_gradients()).await?;
    if saved.len() != retained.len() || saved.iter().zip(&retained).any(|(a, b)| !same_bits(a, b)) {
        return Err("later backwards changed retained gradients".into());
    }
    let before = a.parameter_snapshot();
    let old = read_many(a.tensor_device(), before.values()).await?;
    let update = a.sgd(&bad, 0.)?;
    let receipt = update.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let status = receipt.read_async().await;
    #[cfg(not(target_arch = "wasm32"))]
    let status = receipt.read();
    if !matches!(status, Err(TrainingError::Rejected { stage: 0, flags }) if flags & st_backend_wgpu::resident_tensor::INVALID_TENSOR_FLAG != 0)
    {
        return Err("decoder rejected for the wrong reason or accepted an invalid seed".into());
    }
    let current = a.parameter_snapshot();
    let now = read_many(a.tensor_device(), current.values()).await?;
    if old.len() != now.len() || old.iter().zip(&now).any(|(a, b)| !same_bits(a, b)) {
        return Err("partial decoder update".into());
    }
    if !matches!(
        a.sgd(&good, 0.1),
        Err(InferenceError::Training(TrainingError::ParameterVersion))
    ) {
        return Err("stale whole-model gradients accepted".into());
    }
    if !matches!(
        a.backward(&a2, &good_seed),
        Err(InferenceError::Training(TrainingError::StaleForward))
    ) {
        return Err("tape survived attempted decoder update".into());
    }
    let recovered = a.forward(&batch)?;
    close(
        &read(recovered.prediction()).await?,
        &frozen,
        "rebind after rejection",
    )?;
    Ok(
        json!({"foreign_tape": true, "superseded_tape": true, "retained_output": true, "invalid_bias_preserves_tape": true,
        "good_bad_good": true, "retained_gradients": true, "atomic_rejection": true, "stale_gradient": true, "recovery": true}),
    )
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/resident_byte_decoder_torch.json"
    ))?;
    if fixture["schema"] != "spiraltorch.resident_byte_decoder.torch_fixture.v1"
        || fixture["cases"].as_array().map(Vec::len) != Some(2)
    {
        return Err("incomplete byte decoder oracle".into());
    }
    let geometry_fixture: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/resident_byte_geometry_torch.json"
    ))?;
    if geometry_fixture["schema"] != "spiraltorch.resident_byte_geometry.torch_fixture.v1"
        || geometry_fixture["cases"].as_array().map(Vec::len) != Some(2)
        || geometry_fixture["tolerance"]["geometry_relative_l2"] != 0.002
    {
        return Err("incomplete causal geometry oracle".into());
    }
    let cases: Vec<_> = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(geometry_fixture["cases"].as_array().unwrap())
        .cloned()
        .collect();
    run_cases(runtime, &cases, &[18, 31, 23, 37], false).await
}

/// A local, independent Torch fixture; raw observations are returned only for
/// the new metric control, never added to previously published v3 reports.
pub async fn run_flat_metric(runtime: WgpuRuntime, fixture_json: &str) -> Result<Value> {
    let fixture: Value = serde_json::from_str(fixture_json)?;
    if fixture["schema"] != "spiraltorch.resident_byte_geometry_flat.torch_fixture.v1"
        || fixture["tolerance"] != json!({"atol":3e-6,"rtol":5e-5,"geometry_relative_l2":0.002})
    {
        return Err("invalid flat metric reference contract".into());
    }
    let cases = fixture["cases"]
        .as_array()
        .ok_or("missing flat metric cases")?;
    if cases.len() != 2
        || cases.iter().any(|case| {
            case["config"]["causal_geometry"]["pair_metric"] != "euclidean_chord_squared.v1"
        })
    {
        return Err("incomplete or mislabeled flat metric cases".into());
    }
    run_cases(runtime, cases, &[23, 37], true).await
}

async fn run_cases(
    runtime: WgpuRuntime,
    cases: &[Value],
    counts: &[usize],
    raw: bool,
) -> Result<Value> {
    let mut checks = Vec::new();
    for (case, &expected_count) in cases.iter().zip(counts) {
        let p = plan(case, 4)?;
        if p.parameter_layout().len() != expected_count
            || case["parameter_gradients"].as_array().map(Vec::len) != Some(expected_count)
        {
            return Err("incomplete named parameter oracle".into());
        }
        let windows: Vec<Vec<u8>> = case["windows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                sizes(row)
                    .into_iter()
                    .map(|v| u8::try_from(v).unwrap())
                    .collect()
            })
            .collect();
        let host = batch(&windows, 4)?;
        if host
            .input_bytes()
            .iter()
            .map(|&v| usize::from(v))
            .collect::<Vec<_>>()
            != sizes(&case["inputs"])
            || host
                .target_bytes()
                .iter()
                .map(|&v| usize::from(v))
                .collect::<Vec<_>>()
                != sizes(&case["targets"])
        {
            return Err("byte target alignment changed".into());
        }
        let mut model = p.compile_training_wgpu(runtime.clone())?;
        let resident = model.prepare_batch(&host)?;
        let geometry = biases(model.tensor_device(), case, 4)?;
        let forward = model.forward_with_external_biases(&resident, &borrowed(&geometry))?;
        let output_values = read(forward.prediction()).await?;
        let output_error = close(
            &output_values,
            &data(&case["output"]),
            "byte decoder logits",
        )?;
        let seed_values = data(&case["cotangent"]);
        // Column-major-in-time seed view exercises head/chain packing.
        let mut seed_storage = vec![0.; 2 * 256 * 4];
        for b in 0..2 {
            for t in 0..4 {
                for c in 0..256 {
                    seed_storage[(b * 256 + c) * 4 + t] = seed_values[(b * 4 + t) * 256 + c];
                }
            }
        }
        let seed = model
            .tensor_device()
            .upload(&[2, 256, 4], &seed_storage)?
            .permute(&[0, 2, 1])?;
        let gradient = model.backward(&forward, &seed)?;
        let values = read_many(model.tensor_device(), gradient.parameter_gradients()).await?;
        if values.len() != expected_count {
            return Err("truncated byte decoder gradients".into());
        }
        let parameters = case["parameters"].as_array().unwrap();
        let mut errors = Vec::new();
        for (i, value) in values.iter().enumerate() {
            if gradient.parameter_gradients()[i].layout().shape() != sizes(&parameters[i]["shape"])
            {
                return Err("parameter gradient shape drift".into());
            }
            errors.push(json!({"name": parameters[i]["name"], "max_abs_error": close(value, &data(&case["parameter_gradients"][i]), parameters[i]["name"].as_str().unwrap())?}));
            if p.parameter_layout()
                .geometry()
                .is_some_and(|g| g.all().contains(&i))
            {
                let relative = geometry_controls::relative_gradient(
                    value,
                    &data(&case["parameter_gradients"][i]),
                )?;
                errors.last_mut().unwrap()["relative_l2_error"] = json!(relative);
            }
            if parameters[i]["name"]
                .as_str()
                .unwrap()
                .ends_with("topos_gate")
                && !value.iter().any(|v| v.abs() > 1e-8)
            {
                return Err("Topos gate gradient is inert".into());
            }
        }
        if gradient.embedding_output_gradient().layout().shape() != p.input_layout().shape() {
            return Err("embedding output gradient shape drift".into());
        }
        let input_values = read(gradient.embedding_output_gradient()).await?;
        let input_error = close(
            &input_values,
            &data(&case["embedding_output_gradient"]),
            "embedding output gradient",
        )?;
        if gradient.bias_gradients().len() != p.block_count() {
            return Err("incomplete block bias gradients".into());
        }
        let mut bias_errors = Vec::new();
        let mut bias_values = Vec::new();
        let batch_size = case["config"]["batch"].as_u64().unwrap() as usize;
        let heads = case["config"]["heads"].as_u64().unwrap() as usize;
        for (block, bias) in gradient.bias_gradients().iter().enumerate() {
            for (g, name, shape) in [
                (bias.z_bias(), "z", vec![batch_size, heads, 4]),
                (bias.pair_bias(), "pair", vec![batch_size, heads, 4, 4]),
            ] {
                if g.is_some() == case["biases"][block][name].is_null() {
                    return Err("external bias gradient presence drift".into());
                }
                let Some(g) = g else { continue };
                if g.layout().shape() != shape {
                    return Err("external bias gradient shape drift".into());
                }
                let value = read(g).await?;
                bias_errors.push(close(
                    &value,
                    &data(&case["bias_gradients"][bias_errors.len()]),
                    "external bias gradient",
                )?);
                bias_values.push(value);
            }
        }
        if bias_errors.len() != case["bias_gradients"].as_array().unwrap().len() {
            return Err("missing geometry VJP".into());
        }
        let causal = causal_checks(&runtime, case).await?;
        let tapes = tape_checks(&runtime, case).await?;
        let metric_controls = if p.parameter_layout().geometry().is_some() {
            Some(geometry_controls::run(&runtime, case, &output_values, &input_values).await?)
        } else {
            None
        };
        let learning = &case["learning"];
        if learning["steps"] != 16 || learning["trace"].as_array().map(Vec::len) != Some(16) {
            return Err("incomplete byte learning oracle".into());
        }
        let mut pending = Vec::new();
        for _ in 0..16 {
            let forward = model.forward_with_external_biases(&resident, &borrowed(&geometry))?;
            let loss =
                forward.next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?)?;
            let gradient = model.backward(&forward, loss.prediction_gradient())?;
            let update = model.sgd(&gradient, learning["rate"].as_f64().unwrap() as f32)?;
            let geometry_gradients = p
                .parameter_layout()
                .geometry()
                .map(|g| gradient.parameter_gradients()[g.all()].to_vec())
                .unwrap_or_default();
            pending.push((
                loss,
                update,
                model.parameter_snapshot(),
                geometry_gradients,
                raw.then(|| gradient.embedding_output_gradient().clone()),
            ));
        }
        // No activation/gradient/receipt read until every update is queued.
        let mut trace = Vec::new();
        for (step, (loss, update, parameters, geometry_gradients, embedded)) in
            pending.into_iter().enumerate()
        {
            let receipt = update.snapshot()?;
            #[cfg(target_arch = "wasm32")]
            let revision = receipt.read_async().await?;
            #[cfg(not(target_arch = "wasm32"))]
            let revision = receipt.read()?;
            if revision != (step + 1) as u64 {
                return Err("decoder revision mismatch".into());
            }
            let loss_values = read(loss.value()).await?;
            close(
                &loss_values,
                &[learning["trace"][step]["loss"].as_f64().unwrap() as f32],
                "byte CE",
            )?;
            let actual = read_many(model.tensor_device(), parameters.values()).await?;
            let expected = learning["trace"][step]["parameters"].as_array().unwrap();
            if actual.len() != expected_count || expected.len() != expected_count {
                return Err("truncated decoder update trajectory".into());
            }
            let mut maximum = 0f64;
            for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
                maximum = maximum.max(close(a, &data(e), &format!("step {step} parameter {i}"))?);
            }
            let mut geometry_relative = Vec::new();
            let geometry_values = if geometry_gradients.is_empty() {
                Vec::new()
            } else {
                read_many(model.tensor_device(), &geometry_gradients).await?
            };
            if !geometry_gradients.is_empty() {
                let expected = learning["trace"][step]["geometry_gradients"]
                    .as_array()
                    .ok_or("missing geometry CE derivatives")?;
                if geometry_values.len() != expected.len() {
                    return Err("truncated geometry CE derivatives".into());
                }
                for (a, e) in geometry_values.iter().zip(expected) {
                    close(a, &data(e), "geometry CE derivative")?;
                    geometry_relative.push(geometry_controls::relative_gradient(a, &data(e))?);
                }
            }
            trace.push(json!({"revision": revision, "loss": loss_values[0], "parameter_max_abs_error": maximum,
                "geometry_gradient_relative_l2": geometry_relative}));
            if let Some(embedded) = embedded {
                let row = trace.last_mut().unwrap();
                row["parameters"] = json!(actual);
                row["geometry_gradients"] = json!(geometry_values);
                row["embedding_output_gradient"] = json!(read(&embedded).await?);
            }
        }
        let checkpoint = checkpoint_controls::run(&runtime, case).await?;
        checks.push(json!({"name": case["name"], "checkpoint":checkpoint, "parameter_count": expected_count, "output_max_abs_error": output_error,
            "embedding_output_max_abs_error": input_error, "parameter_errors": errors, "bias_errors": bias_errors, "gradient_layouts": true,
            "causality": causal, "tapes": tapes, "metric_controls": metric_controls, "learning": {"steps":16, "trace":trace}}));
        if raw {
            let check = checks.last_mut().unwrap();
            check["metric"] = json!("euclidean_chord_squared.v1");
            check["parameter_names"] =
                json!(parameters.iter().map(|p| &p["name"]).collect::<Vec<_>>());
            check["parameter_shapes"] =
                json!(parameters.iter().map(|p| &p["shape"]).collect::<Vec<_>>());
            check["output"] = json!(output_values);
            check["parameter_gradients"] = json!(values);
            check["embedding_output_gradient"] = json!(input_values);
            check["bias_gradients"] = json!(bias_values);
            check["resume_trajectory"] = checkpoint_controls::resume_trajectory(
                &runtime,
                case,
                check["learning"]["trace"].as_array().unwrap(),
            )
            .await?;
        }
    }
    Ok(
        json!({"schema":if raw { "spiraltorch.resident_byte_geometry_flat.validation.v1" }
            else { "spiraltorch.resident_byte_decoder.validation.v3" }, "passed":true,
        "adapter":format!("{:?}",runtime.adapter_info()), "checks":checks,
        "scope":"complete 256-way byte decoder correctness and synthetic training, not language quality or speed"}),
    )
}
