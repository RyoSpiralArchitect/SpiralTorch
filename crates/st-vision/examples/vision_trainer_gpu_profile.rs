//! Capture existing VJP GPU passes during the same real-image Rust trainer step.
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use st_backend_wgpu::{resident_tensor::TensorDevice, runtime::WgpuRuntime};
use st_tensor::Tensor;
use st_vision::resident_trainer::{ResidentVisionTrainer, VisionTrainingCheckpoint};
use st_vision::{
    dataset_catalog, DatasetSample, ImageTensor, Normalize, TensorVisionDataset,
    TransformOperation, TransformPipeline,
};
use std::{error::Error, fs, io::Write, path::Path, sync::Arc, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    pixels_file: String,
    pixels_sha256: String,
    labels: Vec<u32>,
    ids: Vec<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    dataset: Input,
    dataset_id: String,
    initial_file: String,
    initial_sha256: String,
    expected_final_file: String,
    expected_final_sha256: String,
    seed: u64,
    steps: usize,
    warmup: usize,
    profile_first: bool,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn checked_file(path: &str, digest: &str) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    if sha(&bytes) != digest {
        return Err("input file hash mismatch".into());
    }
    Ok(bytes)
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(bytes)?;
    Ok(())
}

fn dataset(case: &Case) -> Result<Arc<TensorVisionDataset>> {
    let pixels = checked_file(&case.dataset.pixels_file, &case.dataset.pixels_sha256)?;
    let n = case.dataset.labels.len();
    if n != case.dataset.ids.len() || n == 0 || pixels.len() != n * 3 * 32 * 32 {
        return Err("dataset coverage differs".into());
    }
    let mut identity = Sha256::new();
    identity.update(b"spiraltorch.cifar_trainer_input.v1\0");
    let mut samples = Vec::new();
    for ((image, &target), &id) in pixels
        .chunks_exact(3072)
        .zip(&case.dataset.labels)
        .zip(&case.dataset.ids)
    {
        if target >= 10 || id >= 50_000 {
            return Err("invalid CIFAR label/id".into());
        }
        let values: Vec<f32> = image.iter().map(|&v| f32::from(v) / 255.).collect();
        let label = id.to_string();
        for value in &values {
            identity.update(value.to_le_bytes());
        }
        identity.update((target as f32).to_le_bytes());
        identity.update(label.as_bytes());
        identity.update(b"\0");
        samples.push(
            DatasetSample::new(ImageTensor::new(3, 32, 32, values)?)
                .with_label(label)
                .with_target(Tensor::from_vec(1, 1, vec![target as f32])?),
        );
    }
    if format!("{:x}", identity.finalize()) != case.dataset_id {
        return Err("Rust input identity differs".into());
    }
    let descriptor = dataset_catalog()
        .iter()
        .find(|d| d.name == "CIFAR10")
        .ok_or("missing CIFAR descriptor")?;
    Ok(Arc::new(TensorVisionDataset::from_samples(
        descriptor.clone(),
        samples,
    )?))
}

fn run(
    device: &TensorDevice,
    dataset: Arc<TensorVisionDataset>,
    initial: &VisionTrainingCheckpoint,
    case: &Case,
    count: usize,
    profile: bool,
) -> Result<(Value, String)> {
    let mut pipeline = TransformPipeline::with_seed(case.seed);
    pipeline.add(TransformOperation::Normalize(Normalize::new(
        vec![0.5; 3],
        vec![0.25; 3],
    )?));
    let mut trainer = ResidentVisionTrainer::from_dataset_checkpoint(
        device.clone(),
        dataset,
        Some(pipeline),
        &case.dataset_id,
        initial,
    )?;
    if trainer.checkpoint_snapshot()?.read()?.to_json()? != initial.to_json()? {
        return Err("initial checkpoint changed during restore".into());
    }
    let mut steps = Vec::new();
    for index in 0..count {
        let begin = Instant::now();
        let (submitted, pending) = if profile {
            let (submitted, pending) =
                device.profile_convolution_vjps(|| -> Result<_> { Ok(trainer.submit_next()?) })?;
            (submitted, Some(pending))
        } else {
            (trainer.submit_next()?, None)
        };
        let outcome = trainer.settle()?;
        let elapsed_ns = u64::try_from(begin.elapsed().as_nanos())?;
        if !outcome.accepted || outcome.attempted_revision != (index + 1) as u64 {
            return Err("rejected or incorrect update".into());
        }
        // Query resolve/maps follow completion and are outside the host step clock.
        // This is diagnostic sampling with inter-step observation, not throughput.
        let gpu = pending
            .map(|p| p.read().map(|r| r.to_json_value()))
            .transpose()?;
        steps.push(json!({"step": index + 1, "host_step_ns": elapsed_ns,
            "labels": submitted.labels, "gpu": gpu}));
    }
    let final_json = trainer.checkpoint_snapshot()?.read()?.to_json()?;
    Ok((
        json!({"profiled": profile, "steps": steps, "final_sha256": sha(final_json.as_bytes())}),
        final_json,
    ))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: vision_trainer_gpu_profile CASE_JSON NEW_OUTPUT_DIR".into());
    }
    let case_bytes = fs::read(&args[1])?;
    let case: Case = serde_json::from_slice(&case_bytes)?;
    if case.steps == 0 || case.warmup == 0 {
        return Err("nonzero step/warmup counts required".into());
    }
    let initial_bytes = checked_file(&case.initial_file, &case.initial_sha256)?;
    let initial = VisionTrainingCheckpoint::from_json(std::str::from_utf8(&initial_bytes)?)?;
    let expected = checked_file(&case.expected_final_file, &case.expected_final_sha256)?;
    let initial_value: Value = serde_json::from_slice(&initial_bytes)?;
    let batch = initial_value["input"]["batch_size"]
        .as_u64()
        .ok_or("missing batch size")? as usize;
    if batch == 0
        || case
            .steps
            .max(case.warmup)
            .checked_mul(batch)
            .is_none_or(|n| n > case.dataset.labels.len())
    {
        return Err("profile must remain inside the first epoch".into());
    }
    let dataset = dataset(&case)?;
    let runtime = WgpuRuntime::request_profiled_headless_blocking("vision.trainer.profile")?;
    if runtime.adapter_info().device_type == wgpu::DeviceType::Cpu {
        return Err("GPU required".into());
    }
    let adapter = format!("{:?}", runtime.adapter_info());
    let device = TensorDevice::new(runtime)?;
    let output = Path::new(&args[2]);
    fs::create_dir(output)?;
    let mut records = Vec::new();
    for profiled in [case.profile_first, !case.profile_first] {
        run(
            &device,
            dataset.clone(),
            &initial,
            &case,
            case.warmup,
            profiled,
        )?;
        let (record, final_json) = run(
            &device,
            dataset.clone(),
            &initial,
            &case,
            case.steps,
            profiled,
        )?;
        write_new(
            &output.join(if profiled {
                "profiled-final.json"
            } else {
                "control-final.json"
            }),
            final_json.as_bytes(),
        )?;
        if final_json.as_bytes() != expected {
            return Err("full checkpoint differs from retained matched run".into());
        }
        records.push(record);
    }
    let result = json!({"schema": "spiraltorch.vision.trainer_gpu_profile.v1", "passed": true,
        "boundary": "Diagnostic convolution VJP pass sampling; inter-step query reads; not throughput or all GPU work",
        "adapter": adapter, "case_sha256": sha(&case_bytes), "dataset_id": case.dataset_id,
        "initial_sha256": case.initial_sha256, "expected_final_sha256": case.expected_final_sha256,
        "exact_retained_checkpoint": true, "batch_size": batch, "records": records});
    write_new(
        &output.join("result.json"),
        &serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{{\"passed\":true,\"batch_size\":{batch},\"exact_retained_checkpoint\":true}}");
    Ok(())
}
