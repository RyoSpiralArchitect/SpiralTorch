//! Read-only, fixed-batch gradient localization on a request-bound saved study.
use serde_json::{json, Value};
use st_backend_wgpu::runtime;
use st_kernel_contracts::classification::{ClassReduction, CrossEntropySpec};
use st_nn::resident::{ByteCorpusStudy, ByteLmBatch, BYTE_CORPUS_STUDY_MAX_BYTES};
use std::io::Read;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn read(path: &Path) -> Result<Vec<u8>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > BYTE_CORPUS_STUDY_MAX_BYTES as u64 {
        return Err("expected a regular file of at most 64 MiB".into());
    }
    let mut bytes = Vec::new();
    file.take(BYTE_CORPUS_STUDY_MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
        return Err("file grew beyond 64 MiB".into());
    }
    Ok(bytes)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: resident_byte_gradient_probe REQUEST CHECKPOINT".into());
    }
    let request = read(Path::new(&args[0]))?;
    let study = ByteCorpusStudy::from_json(&request)?;
    let checkpoint = study.checkpoint_from_json(&read(Path::new(&args[1]))?)?;
    if study.total_updates() != 1024 || checkpoint.completed_updates() != 1024 {
        return Err("this bounded probe requires the complete 1024-update study".into());
    }
    let value: Value = serde_json::from_slice(&request)?;
    let documents: Vec<Vec<u8>> = serde_json::from_value(value["train_documents"].clone())?;
    let selections: Vec<Vec<(usize, usize)>> =
        serde_json::from_value(value["train_batches"].clone())?;
    let steps = value["config"]["steps"].as_u64().ok_or("missing steps")? as usize;
    let docs: Vec<_> = documents.iter().map(Vec::as_slice).collect();
    let (runtime, _) = runtime::ensure_default_runtime_blocking("byte.gradient.probe")?;
    if format!("{:?}", runtime.adapter_info().device_type) == "Cpu" {
        return Err("real GPU required".into());
    }
    let mut cases = Vec::new();
    for case in value["cases"].as_array().ok_or("missing cases")? {
        let name = case["name"].as_str().ok_or("missing case name")?;
        let mut model = study
            .case_checkpoint(&checkpoint, name)?
            .restore_wgpu(runtime.clone())?;
        let before = model.checkpoint_snapshot()?.read()?.to_json()?;
        let parameters = model
            .parameter_snapshot()
            .values()
            .iter()
            .map(|p| p.snapshot()?.read())
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut probes = Vec::new();
        for index in [0usize, 127, 511, 1023] {
            let batch = ByteLmBatch::from_documents(&docs, &selections[index], steps)?;
            let batch = model.prepare_batch(&batch)?;
            let forward = model.forward(&batch)?;
            let loss =
                forward.next_byte_loss(CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)?)?;
            let gradients = model.backward(&forward, loss.prediction_gradient())?;
            let values = gradients
                .parameter_gradients()
                .iter()
                .map(|p| p.snapshot()?.read())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            probes.push(json!({"batch_index": index,
                "loss": loss.value().snapshot()?.read()?[0], "gradients": values}));
        }
        if before != model.checkpoint_snapshot()?.read()?.to_json()? {
            return Err("read-only probe changed parameters, moments or revision".into());
        }
        cases.push(
            json!({"name": name, "parameter_values": parameters, "probes": probes,
                         "checkpoint_unchanged": true}),
        );
    }
    println!(
        "{}",
        serde_json::to_string(&json!({
            "schema": "spiraltorch.byte_gradient_probe.v1",
            "request_sha256": study.request_sha256(),
            "adapter": format!("{:?}", runtime.adapter_info()),
            "cases": cases,
            "scope": "fixed training batches; no update, selection or language-quality claim"
        }))?
    );
    Ok(())
}
