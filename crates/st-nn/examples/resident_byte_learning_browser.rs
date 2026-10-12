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
