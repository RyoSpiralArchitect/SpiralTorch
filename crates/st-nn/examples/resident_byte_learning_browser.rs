#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[path = "resident_byte_learning_browser/serving.rs"]
mod serving;

/// CPU-only selection. Transport an array of opaque checkpoint JSON strings;
/// never parse/reconstruct their inner model numbers in JavaScript.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn select_resident_byte_learning_checkpoint(
    input: &str,
    checkpoints: Vec<String>,
    case_name: &str,
    policy: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    use st_nn::resident::{ByteCorpusStudy, BYTE_CORPUS_STUDY_MAX_BYTES};
    let error = |e: Box<dyn std::error::Error>| wasm_bindgen::JsValue::from_str(&e.to_string());
    if !(1..=32).contains(&checkpoints.len())
        || checkpoints
            .iter()
            .try_fold(0usize, |n, s| n.checked_add(s.len()))
            .is_none_or(|n| n > BYTE_CORPUS_STUDY_MAX_BYTES)
    {
        return Err(wasm_bindgen::JsValue::from_str(
            "selection needs 1..32 checkpoints, at most 64 MiB combined",
        ));
    }
    let study = ByteCorpusStudy::from_json(input.as_bytes()).map_err(error)?;
    let policy = serde_json::from_value(serde_json::Value::String(policy.to_owned()))
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let saved = checkpoints
        .iter()
        .map(|s| study.checkpoint_from_json(s.as_bytes()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(error)?;
    let refs: Vec<_> = saved.iter().collect();
    let selected = study
        .select_checkpoint(&refs, case_name, policy)
        .map_err(error)?;
    Ok(
        serde_json::json!({"artifact_json":selected.artifact.to_json(),
        "report":selected.report})
        .to_string(),
    )
}

/// CPU-only metadata inspection uses the same parser as execution.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn inspect_resident_byte_learning(input: &str) -> Result<String, wasm_bindgen::JsValue> {
    let error = |e: Box<dyn std::error::Error>| wasm_bindgen::JsValue::from_str(&e.to_string());
    let study = st_nn::resident::ByteCorpusStudy::from_json(input.as_bytes()).map_err(error)?;
    let optimizer: serde_json::Value =
        serde_json::from_str(&study.optimizer_recipe_json().map_err(error)?)
            .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    Ok(serde_json::json!({
        "schema":"spiraltorch.byte_corpus.inspection.v1",
        "request_sha256":study.request_sha256(),
        "design_schema":study.design_schema(),
        "optimizer":optimizer,
        "total_updates":study.total_updates(),
        "checkpoint_size_bound":study.checkpoint_size_bound().map_err(error)?
    })
    .to_string())
}

/// CPU-only, all-case source binding; keep this opaque JSON intact in transport.
#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn pack_resident_byte_learning_case_artifact(
    input: &str,
    checkpoint: &str,
    case_name: &str,
    policy: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    let error = |e: Box<dyn std::error::Error>| wasm_bindgen::JsValue::from_str(&e.to_string());
    let policy = serde_json::from_value(serde_json::Value::String(policy.to_owned()))
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let study = st_nn::resident::ByteCorpusStudy::from_json(input.as_bytes()).map_err(error)?;
    let saved = study
        .checkpoint_from_json(checkpoint.as_bytes())
        .map_err(error)?;
    study
        .case_artifact(&saved, case_name, policy)
        .map(|a| a.to_json().to_owned())
        .map_err(error)
}

/// Recheck the private source companions without a device or new training.
#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn verify_resident_byte_learning_artifact(
    input: &str,
    checkpoint: &str,
    artifact: &str,
    expected_id: &str,
) -> Result<(), wasm_bindgen::JsValue> {
    let error = |e: Box<dyn std::error::Error>| wasm_bindgen::JsValue::from_str(&e.to_string());
    let artifact =
        st_nn::resident::ByteCorpusModelArtifact::from_json(artifact.as_bytes(), expected_id)
            .map_err(error)?;
    let study = st_nn::resident::ByteCorpusStudy::from_json(input.as_bytes()).map_err(error)?;
    let saved = study
        .checkpoint_from_json(checkpoint.as_bytes())
        .map_err(error)?;
    study.validate_artifact(&saved, &artifact).map_err(error)
}

/// Fresh policy-constrained inference, not a promotion gate or streaming decode.
/// Windows contain B rows of T+1 bytes, with the last byte supplying a CE label.
#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn infer_resident_byte_artifact(
    artifact: &str,
    expected_id: &str,
    windows: Vec<u8>,
) -> Result<String, wasm_bindgen::JsValue> {
    use st_kernel_contracts::classification::{ClassReduction, CrossEntropySpec};
    use st_nn::resident::{ByteCorpusModelArtifact, ByteLmBatch};
    let error = |e: Box<dyn std::error::Error>| wasm_bindgen::JsValue::from_str(&e.to_string());
    let artifact =
        ByteCorpusModelArtifact::from_json(artifact.as_bytes(), expected_id).map_err(error)?;
    let shape = artifact.checkpoint().plan().input_layout().shape();
    let row = shape[1]
        .checked_add(1)
        .ok_or_else(|| wasm_bindgen::JsValue::from_str("window overflow"))?;
    if shape[0].checked_mul(row) != Some(windows.len()) {
        return Err(wasm_bindgen::JsValue::from_str(
            "windows differ from artifact shape",
        ));
    }
    let batch = ByteLmBatch::from_windows(&windows.chunks_exact(row).collect::<Vec<_>>())
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let runtime = st_backend_wgpu::runtime::WgpuRuntime::request_headless("byte.artifact.browser")
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let mut model = artifact
        .restore_wgpu(runtime)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let batch = model
        .prepare_batch(&batch)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let tagged = model
        .forward(&batch)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let forward = tagged.forward();
    let spec = CrossEntropySpec::new(ClassReduction::Mean, -100, 0.)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let loss = forward
        .next_byte_loss(spec)
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let capture = model
        .tensor_device()
        .snapshot_many(&[forward.prediction(), loss.value()])
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let values = capture
        .read_async()
        .await
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?;
    let adapter = model.tensor_device().runtime().adapter_info();
    Ok(serde_json::json!({"artifact_id":tagged.artifact_id(), "revision":forward.parameter_revision(),
        "logits":values[0], "loss":values[1][0],
        "adapter":{"name":adapter.name, "backend":format!("{:?}",adapter.backend),
            "device_type":format!("{:?}",adapter.device_type)}}).to_string())
}

/// CPU-only transport to the existing byte-decoder checkpoint contract.
/// The source request and study checkpoint remain the provenance/resume record.
#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn extract_resident_byte_learning_case(
    input: &str,
    checkpoint: &str,
    case_name: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    let error = |e: Box<dyn std::error::Error>| wasm_bindgen::JsValue::from_str(&e.to_string());
    let study = st_nn::resident::ByteCorpusStudy::from_json(input.as_bytes()).map_err(error)?;
    let saved = study
        .checkpoint_from_json(checkpoint.as_bytes())
        .map_err(error)?;
    study
        .case_checkpoint(&saved, case_name)
        .map_err(error)?
        .to_json()
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn prepare_resident_byte_learning_query_mixtures(
    input: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    let prepared = st_nn::resident::ByteCorpusQueryMixtureControls::from_json(input.as_bytes())
        .map_err(|e| wasm_bindgen::JsValue::from_str(&e.to_string()))?
        .prepare();
    Ok(
        serde_json::json!({"schema":"spiraltorch.byte_corpus.prepared.v1",
        "request_json":prepared.request_json,"report_json":prepared.report.to_string()})
        .to_string(),
    )
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn prepare_resident_byte_learning_mixtures(
    input: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    use st_nn::resident::ByteCorpusMixtureControls;
    use wasm_bindgen::JsValue;
    let error = |e: Box<dyn std::error::Error>| JsValue::from_str(&e.to_string());
    let preparation = ByteCorpusMixtureControls::from_json(input.as_bytes()).map_err(error)?;
    let runtime = st_backend_wgpu::runtime::WgpuRuntime::request_headless("byte.corpus.mixture")
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let prepared = preparation.prepare(runtime).await.map_err(error)?;
    Ok(
        serde_json::json!({"schema":"spiraltorch.byte_corpus.prepared.v1",
        "request_json":prepared.request_json,"report_json":prepared.report.to_string()})
        .to_string(),
    )
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn prepare_resident_byte_learning_headwise(
    input: &str,
) -> Result<String, wasm_bindgen::JsValue> {
    use st_nn::resident::ByteCorpusHeadwiseControls;
    use wasm_bindgen::JsValue;
    let error = |e: Box<dyn std::error::Error>| JsValue::from_str(&e.to_string());
    let preparation = ByteCorpusHeadwiseControls::from_json(input.as_bytes()).map_err(error)?;
    let runtime = st_backend_wgpu::runtime::WgpuRuntime::request_headless("byte.corpus.headwise")
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let prepared = preparation.prepare(runtime).await.map_err(error)?;
    Ok(
        serde_json::json!({"schema":"spiraltorch.byte_corpus.prepared.v1",
        "request_json":prepared.request_json,"report_json":prepared.report.to_string()})
        .to_string(),
    )
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn prepare_resident_byte_learning(
    input: &str,
    train_batch_indices: Vec<u32>,
) -> Result<String, wasm_bindgen::JsValue> {
    prepare(input, train_batch_indices, false).await
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn prepare_resident_byte_learning_fisher(
    input: &str,
    train_batch_indices: Vec<u32>,
) -> Result<String, wasm_bindgen::JsValue> {
    prepare(input, train_batch_indices, true).await
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
async fn prepare(
    input: &str,
    train_batch_indices: Vec<u32>,
    fisher: bool,
) -> Result<String, wasm_bindgen::JsValue> {
    use st_nn::resident::ByteCorpusBiasCalibration;
    use wasm_bindgen::JsValue;
    let error = |e: Box<dyn std::error::Error>| JsValue::from_str(&e.to_string());
    let indices: Vec<_> = train_batch_indices
        .into_iter()
        .map(|i| i as usize)
        .collect();
    let mut preparation =
        ByteCorpusBiasCalibration::from_json(input.as_bytes(), &indices).map_err(error)?;
    if fisher {
        preparation = preparation.with_fisher_rao_control().map_err(error)?;
    }
    let runtime =
        st_backend_wgpu::runtime::WgpuRuntime::request_headless("byte.corpus.calibration")
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let prepared = preparation.prepare(runtime).await.map_err(error)?;
    Ok(
        serde_json::json!({"schema":"spiraltorch.byte_corpus.prepared.v1",
        "request_json":prepared.request_json,"report_json":prepared.report.to_string()})
        .to_string(),
    )
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn run_resident_byte_learning(input: &str) -> Result<String, wasm_bindgen::JsValue> {
    use st_backend_wgpu::runtime::WgpuRuntime;
    use wasm_bindgen::JsValue;
    let study = st_nn::resident::ByteCorpusStudy::from_json(input.as_bytes())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let runtime = WgpuRuntime::request_headless("byte.corpus.browser")
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    study
        .run(runtime)
        .await
        .map(|v| v.to_string())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[cfg(all(feature = "wgpu", target_arch = "wasm32"))]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn advance_resident_byte_learning(
    input: &str,
    checkpoint: Option<String>,
    stop_after: u32,
) -> Result<String, wasm_bindgen::JsValue> {
    use st_nn::resident::ByteCorpusStudy;
    use wasm_bindgen::JsValue;
    let error = |e: Box<dyn std::error::Error>| JsValue::from_str(&e.to_string());
    let study = ByteCorpusStudy::from_json(input.as_bytes()).map_err(error)?;
    let resume = checkpoint
        .as_ref()
        .map(|json| study.checkpoint_from_json(json.as_bytes()))
        .transpose()
        .map_err(error)?;
    study
        .validate_segment(resume.as_ref(), stop_after as usize)
        .map_err(error)?;
    let runtime = st_backend_wgpu::runtime::WgpuRuntime::request_headless("byte.corpus.browser")
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let segment = study
        .advance(runtime, resume.as_ref(), stop_after as usize)
        .await
        .map_err(error)?;
    Ok(
        serde_json::json!({"schema":"spiraltorch.byte_corpus.segment.v1",
        "completed_updates":segment.checkpoint.completed_updates(),
        "checkpoint_json":segment.checkpoint.to_json().map_err(error)?,
        "report_json":segment.report.to_string()})
        .to_string(),
    )
}
