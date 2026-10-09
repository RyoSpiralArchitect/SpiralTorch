//! One frozen Torch oracle and guard suite shared by native and browser probes.
use serde_json::{json, Value};
use st_backend_wgpu::{
    resident_tensor::{ResidentTensor, TensorDevice, TensorError, INVALID_TENSOR_FLAG},
    resident_training::{parameters::ResidentParameters, TrainingError},
    runtime::WgpuRuntime,
};
use st_kernel_contracts::{
    classification::{ClassReduction, CrossEntropySpec},
    layout::NdLayout,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn sizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

async fn read(tensor: &ResidentTensor) -> Result<Vec<f32>> {
    let snapshot = tensor.snapshot()?;
    #[cfg(target_arch = "wasm32")]
    let values = snapshot.read_async().await?;
    #[cfg(not(target_arch = "wasm32"))]
    let values = snapshot.read()?;
    Ok(values)
}

async fn rejected(tensor: &ResidentTensor) -> Result<()> {
    if !matches!(read(tensor).await, Err(e) if matches!(e.downcast_ref::<TensorError>(), Some(TensorError::NonFinite)))
    {
        return Err("embedding lost a whole-operation guard".into());
    }
    Ok(())
}

fn require_embedding_rejection(status: std::result::Result<u64, TrainingError>) -> Result<()> {
    match status {
        Err(TrainingError::Rejected { stage: 0, flags }) if flags & INVALID_TENSOR_FLAG != 0 => {
            Ok(())
        }
        Err(error) => Err(error.into()),
        Ok(_) => Err("overflowing embedding update accepted".into()),
    }
}

fn rejection_negative_controls() -> Result<usize> {
    let mut count = 0;
    for status in [
        Ok(1),
        Err(TrainingError::InvalidReadback),
        Err(TrainingError::Rejected {
            stage: 1,
            flags: INVALID_TENSOR_FLAG,
        }),
        Err(TrainingError::Rejected { stage: 0, flags: 1 }),
    ] {
        if require_embedding_rejection(status).is_ok() {
            return Err("unrelated failure counted as embedding rejection".into());
        }
        count += 1;
    }
    require_embedding_rejection(Err(TrainingError::Rejected {
        stage: 0,
        flags: INVALID_TENSOR_FLAG,
    }))?;
    Ok(count)
}

fn compare(label: &str, actual: &[f32], expected: &[f32]) -> Result<f64> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: length mismatch {} != {}",
            actual.len(),
            expected.len()
        )
        .into());
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

fn upload(
    device: &TensorDevice,
    shape: &[usize],
    values: &[f32],
    strided: bool,
) -> Result<ResidentTensor> {
    if !strided {
        return Ok(device.upload(shape, values)?);
    }
    let axes: Vec<_> = (0..shape.len()).rev().collect();
    let mut padded: Vec<_> = shape.iter().rev().copied().collect();
    padded[0] += 2;
    let storage = NdLayout::contiguous(&padded)?;
    let view = storage
        .narrow(0, 1, shape[shape.len() - 1])?
        .permute(&axes)?;
    let mut data = vec![0.; storage.len()];
    for (i, &value) in values.iter().enumerate() {
        data[view.storage_index(i).ok_or("fixture length")?] = value;
    }
    Ok(device
        .upload(&padded, &data)?
        .narrow(0, 1, shape[shape.len() - 1])?
        .permute(&axes)?)
}

async fn edge_checks(device: &TensorDevice) -> Result<Value> {
    let rejection_negative_controls = rejection_negative_controls()?;
    let mut rejected_shapes = 0;
    for (shape, rows, ids) in [
        (vec![1], 2, vec![2]),
        (vec![1], 2, vec![usize::MAX]),
        (vec![0], 2, vec![0]),
        (vec![], 2, vec![]),
        (vec![1], 0, vec![0]),
        (vec![0], usize::MAX, vec![]),
    ] {
        if device.upload_embedding_indices(&shape, rows, &ids).is_ok() {
            return Err("invalid embedding index plan accepted".into());
        }
        rejected_shapes += 1;
    }
    let ids = device.upload_embedding_indices(&[1], 2, &[0])?;
    for shape in [vec![2], vec![3, 1], vec![1, 2, 1]] {
        if device
            .upload(&shape, &vec![0.; shape.iter().product()])?
            .embedding(&ids)
            .is_ok()
        {
            return Err("invalid embedding table shape accepted".into());
        }
        rejected_shapes += 1;
    }
    let table = device.upload(&[2, 1], &[1., 2.])?;
    let forward = table.embedding(&ids)?;
    if forward.backward(&device.upload(&[1], &[1.])?).is_ok() {
        return Err("flattened embedding cotangent accepted".into());
    }
    rejected_shapes += 1;

    // Failure is outside the selected row. Empty lookups must also retain it.
    let large = device.upload(&[2, 1], &[0., f32::MAX])?;
    let failed = large.mul(&device.upload(&[2, 1], &[1., 2.])?)?;
    let alias = device.guard_together(&[&table, &failed])?.remove(0);
    let mut guard_cases = 0;
    for source in [&failed, &alias] {
        for n in [0, 1] {
            let ids = device.upload_embedding_indices(&[n], 2, &vec![0; n])?;
            let forward = source.embedding(&ids)?;
            rejected(forward.prediction()).await?;
            rejected(&forward.backward(&device.upload(&[n, 1], &vec![1.; n])?)?).await?;
            guard_cases += 1;
        }
    }
    let bad_seed = failed.narrow(0, 0, 1)?;
    rejected(&forward.backward(&bad_seed)?).await?;
    let aliased_seed = device
        .guard_together(&[&device.upload(&[1, 1], &[1.])?, &failed])?
        .remove(0);
    rejected(&forward.backward(&aliased_seed)?).await?;
    guard_cases += 2;

    // Broadcast table and cotangent: lookup is logical, not storage indexing.
    let broadcast = device
        .upload(&[1, 3], &[1., -0., 3.])?
        .broadcast_to(&[5, 3])?;
    let broadcast_ids = device.upload_embedding_indices(&[2, 3], 5, &[4, 1, 4, 1, 0, 4])?;
    let tape = broadcast.embedding(&broadcast_ids)?;
    let expected: Vec<_> = (0..6).flat_map(|_| [1., -0., 3.]).collect();
    if !read(tape.prediction())
        .await?
        .iter()
        .map(|x| x.to_bits())
        .eq(expected.iter().map(|x: &f32| x.to_bits()))
    {
        return Err("embedding gather changed selected value bits".into());
    }
    let seed = device
        .upload(&[1, 1, 3], &[1., 2., 3.])?
        .broadcast_to(&[2, 3, 3])?;
    compare(
        "broadcast VJP",
        &read(&tape.backward(&seed)?).await?,
        &[1., 2., 3., 2., 4., 6., 0., 0., 0., 0., 0., 0., 3., 6., 9.],
    )?;

    // Accumulation order and subnormals are an integer-defined f32 contract.
    let repeat = device.upload_embedding_indices(&[3], 2, &[0, 0, 0])?;
    let tape = table.embedding(&repeat)?;
    let ordered = tape.backward(&device.upload(&[3, 1], &[16_777_216., 1., -16_777_216.])?)?;
    compare("stable order", &read(&ordered).await?, &[0., 0.])?;
    let tiny = tape.backward(&device.upload(&[3, 1], &[f32::from_bits(1); 3])?)?;
    let tiny = read(&tiny).await?;
    if tiny.len() != 2 || tiny[0].to_bits() != 3 || tiny[1].to_bits() != 0 {
        return Err("embedding pullback flushed subnormals".into());
    }
    let overflow = tape.backward(&device.upload(&[3, 1], &[f32::MAX, f32::MAX, -f32::MAX])?)?;
    rejected(&overflow).await?;
    rejected(&overflow.narrow(0, 1, 1)?).await?;
    guard_cases += 1;

    let mut owner = ResidentParameters::new(vec![table.clone(), device.upload(&[1, 1], &[7.])?])?;
    let before = owner.snapshot();
    let gradients = vec![overflow, device.upload(&[1, 1], &[1.])?];
    let mut atomic_rejections = 0;
    for rate in [0., 0.125] {
        let version = owner.snapshot();
        let gradient = version.bind_gradients(gradients.clone())?;
        let update = owner.sgd(&gradient, rate)?;
        let snapshot = update.snapshot()?;
        #[cfg(target_arch = "wasm32")]
        let status = snapshot.read_async().await;
        #[cfg(not(target_arch = "wasm32"))]
        let status = snapshot.read();
        require_embedding_rejection(status)?;
        let after = owner.snapshot();
        if after.values().len() != 2 {
            return Err("truncated parameter snapshot".into());
        }
        for (old, new) in before.values().iter().zip(after.values()) {
            let old = read(old).await?;
            let new = read(new).await?;
            if old.len() != new.len()
                || !old
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(new.iter().map(|x| x.to_bits()))
            {
                return Err("embedding overflow partially changed parameters".into());
            }
        }
        if owner.sgd(&gradient, rate).is_ok() {
            return Err("stale embedding gradient accepted".into());
        }
        atomic_rejections += 1;
    }

    let second =
        TensorDevice::new(WgpuRuntime::request_headless("embedding.foreign_device").await?)?;
    let foreign = second.upload_embedding_indices(&[1], 2, &[0])?;
    if !matches!(table.embedding(&foreign), Err(TensorError::DeviceMismatch))
        || !matches!(
            forward.backward(&second.upload(&[1, 1], &[1.])?),
            Err(TensorError::DeviceMismatch)
        )
    {
        return Err("cross-device embedding accepted".into());
    }
    Ok(
        json!({"rejected_shapes": rejected_shapes, "guard_cases": guard_cases,
              "rejection_negative_controls": rejection_negative_controls,
              "atomic_rejections": atomic_rejections, "cross_device_rejections": 2,
              "broadcast": true, "stable_order": true, "subnormals": true}),
    )
}

pub async fn run(runtime: WgpuRuntime) -> Result<Value> {
    let adapter = format!("{:?}", runtime.adapter_info());
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/resident_embedding_torch.json"))?;
    if fixture["schema"] != "spiraltorch.resident_embedding.torch_fixture.v1"
        || fixture["cases"].as_array().map(Vec::len) != Some(6)
    {
        return Err("incomplete embedding fixture".into());
    }
    let device = TensorDevice::new(runtime)?;
    let mut checks = Vec::new();
    for strided in [false, true] {
        for case in fixture["cases"].as_array().unwrap() {
            let table_shape = sizes(&case["table_shape"]);
            let output_shape = sizes(&case["output_shape"]);
            let indices = device.upload_embedding_indices(
                &sizes(&case["index_shape"]),
                table_shape[0],
                &sizes(&case["indices"]),
            )?;
            let table = upload(&device, &table_shape, &floats(&case["table"]), strided)?;
            let cotangent = upload(&device, &output_shape, &floats(&case["cotangent"]), strided)?;
            let forward = table.embedding(&indices)?;
            if forward.prediction().layout().shape() != output_shape {
                return Err("embedding output axes changed".into());
            }
            let gradient = forward.backward(&cotangent)?;
            let repeated = forward.backward(&cotangent)?;
            if gradient.layout().shape() != table_shape {
                return Err("embedding table VJP shape changed".into());
            }
            drop((table, indices, cotangent));
            let output_error = compare(
                "embedding output",
                &read(forward.prediction()).await?,
                &floats(&case["output"]),
            )?;
            let actual = read(&gradient).await?;
            let gradient_error = compare("embedding VJP", &actual, &floats(&case["gradient"]))?;
            let repeated = read(&repeated).await?;
            if actual.len() != repeated.len()
                || !actual
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(repeated.iter().map(|x| x.to_bits()))
            {
                return Err("non-deterministic embedding pullback".into());
            }
            checks.push(json!({"name": case["name"], "strided": strided,
                "output_max_abs_error": output_error, "gradient_max_abs_error": gradient_error}));
        }
    }
    let edges = edge_checks(&device).await?;
    let learning = &fixture["learning"];
    let steps = learning["steps"].as_u64().unwrap() as usize;
    if steps != 16 || learning["trace"].as_array().map(Vec::len) != Some(steps) {
        return Err("incomplete embedding learning fixture".into());
    }
    let index_shape = sizes(&learning["index_shape"]);
    let table_shape = sizes(&learning["table_shape"]);
    let indices = device.upload_embedding_indices(
        &index_shape,
        table_shape[0],
        &sizes(&learning["indices"]),
    )?;
    let targets = device.upload(&index_shape, &floats(&learning["target"]))?;
    let mut owner = ResidentParameters::new(vec![
        device.upload(&table_shape, &floats(&learning["initial"]))?
    ])?;
    let initial = owner.snapshot();
    let mut pending = Vec::new();
    for _ in 0..steps {
        let snapshot = owner.snapshot();
        let tape = snapshot.values()[0].embedding(&indices)?;
        let loss = tape.prediction().cross_entropy_with_logits(
            &targets,
            CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?,
        )?;
        let gradient = tape.backward(loss.prediction_gradient())?;
        let bound = snapshot.bind_gradients(vec![gradient.clone()])?;
        let update = owner.sgd(&bound, learning["learning_rate"].as_f64().unwrap() as f32)?;
        pending.push((tape, loss, gradient, update));
    }
    // All steps submit before any explicit readback; saved tapes stay immutable.
    let mut trace = Vec::new();
    for (step, (tape, loss, gradient, update)) in pending.into_iter().enumerate() {
        let receipt = update.snapshot()?;
        #[cfg(target_arch = "wasm32")]
        let revision = receipt.read_async().await?;
        #[cfg(not(target_arch = "wasm32"))]
        let revision = receipt.read()?;
        if revision != (step + 1) as u64 {
            return Err("unexpected embedding update revision".into());
        }
        let expected = &learning["trace"][step];
        let logits_error = compare(
            "embedding CE logits",
            &read(tape.prediction()).await?,
            &floats(&expected["logits"]),
        )?;
        let gradient_error = compare(
            "embedding CE VJP",
            &read(&gradient).await?,
            &floats(&expected["gradient"]),
        )?;
        let value = read(loss.value()).await?;
        compare(
            "embedding CE loss",
            &value,
            &[expected["loss"].as_f64().unwrap() as f32],
        )?;
        trace.push(json!({"revision": revision, "loss": value[0], "logits_max_abs_error": logits_error, "gradient_max_abs_error": gradient_error}));
    }
    let final_snapshot = owner.snapshot();
    if final_snapshot.values().len() != 1 {
        return Err("truncated embedding parameters".into());
    }
    let final_error = compare(
        "final embedding table",
        &read(&final_snapshot.values()[0]).await?,
        &floats(&learning["final"]),
    )?;
    compare(
        "initial snapshot unchanged",
        &read(&initial.values()[0]).await?,
        &floats(&learning["initial"]),
    )?;
    Ok(
        json!({"schema": "spiraltorch.resident_embedding.validation.v1", "passed": true,
        "adapter": adapter, "checks": checks, "edge_checks": edges,
        "learning": {"steps": steps, "trace": trace, "final_parameter_max_abs_error": final_error},
        "scope": "embedding correctness and resident CE/SGD, not a language model or speed benchmark"}),
    )
}
