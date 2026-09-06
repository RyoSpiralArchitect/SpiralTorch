use js_sys::{Float32Array, Int32Array, JsString, Number, Promise};
use st_backend_wgpu::{
    rankk_exact_2ce::{resident::ResidentRank, Kind, Plan},
    resident_matmul::{MatmulAccumulation, MatmulKernel, MatmulShape, MatmulTile, ResidentMatmul},
    runtime,
};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::future_to_promise;

fn error(err: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&err.to_string()).into()
}

fn dimension(value: &JsValue) -> Result<usize, JsValue> {
    let value = crate::utils::js_u32(value, "dimension or repetition")?;
    if value == 0 {
        return Err(error("dimension or repetition must be positive"));
    }
    Ok(value as usize)
}

fn float32_array(value: JsValue) -> Result<Float32Array, JsValue> {
    if !crate::utils::js_is_typed_array(&value, "Float32Array")? {
        return Err(error("operand must be a Float32Array"));
    }
    Ok(value.unchecked_into())
}

/// An explicit WebGPU workspace. It never falls back to WASM CPU tensor math.
#[wasm_bindgen(js_name = WgpuMatmul)]
pub struct WasmWgpuMatmul {
    inner: ResidentMatmul,
}

async fn create_workspace(
    shape: MatmulShape,
    tile: MatmulTile,
    kernel: Option<MatmulKernel>,
    accumulation: MatmulAccumulation,
) -> Result<WasmWgpuMatmul, JsValue> {
    let runtime = ensure_runtime().await?;
    Ok(WasmWgpuMatmul {
        inner: ResidentMatmul::with_options(
            runtime,
            shape,
            tile,
            kernel.unwrap_or(MatmulKernel::Scalar),
            accumulation,
        )
        .map_err(error)?,
    })
}

async fn ensure_runtime() -> Result<runtime::WgpuRuntime, JsValue> {
    let runtime = match runtime::default_runtime() {
        Some(runtime) => runtime,
        None => {
            let candidate = runtime::WgpuRuntime::request_headless("wasm.resident.matmul")
                .await
                .map_err(error)?;
            // Concurrent create() promises share whichever runtime installed first.
            let _ = runtime::install_default_runtime(candidate);
            runtime::default_runtime().ok_or_else(|| error("WebGPU runtime installation failed"))?
        }
    };
    Ok(runtime)
}

fn timestamp_request(value: Option<js_sys::Boolean>) -> Result<bool, JsValue> {
    value
        .as_ref()
        .map(|v| {
            v.as_bool()
                .ok_or_else(|| error("timestamp_queries must be a boolean"))
        })
        .transpose()
        .map(|v| v.unwrap_or(false))
}

async fn rank_runtime(timestamps: bool) -> Result<runtime::WgpuRuntime, JsValue> {
    if timestamps {
        runtime::WgpuRuntime::request_profiled_headless("wasm.profiled.rank")
            .await
            .map_err(error)
    } else {
        ensure_runtime().await
    }
}

#[wasm_bindgen(js_class = WgpuMatmul)]
impl WasmWgpuMatmul {
    #[wasm_bindgen(js_name = create)]
    pub async fn create(
        rows: Number,
        inner: Number,
        cols: Number,
    ) -> Result<WasmWgpuMatmul, JsValue> {
        let shape = MatmulShape::new(
            dimension(rows.as_ref())?,
            dimension(inner.as_ref())?,
            dimension(cols.as_ref())?,
        )
        .map_err(error)?;
        create_workspace(
            shape,
            MatmulTile::default(),
            None,
            MatmulAccumulation::Sequential,
        )
        .await
    }

    #[wasm_bindgen(js_name = createWithTile)]
    pub async fn create_with_tile(
        rows: Number,
        inner: Number,
        cols: Number,
        tile_m: Number,
        tile_n: Number,
        tile_k: Number,
    ) -> Result<WasmWgpuMatmul, JsValue> {
        let shape = MatmulShape::new(
            dimension(rows.as_ref())?,
            dimension(inner.as_ref())?,
            dimension(cols.as_ref())?,
        )
        .map_err(error)?;
        let tile = MatmulTile::new(
            dimension(tile_m.as_ref())? as u32,
            dimension(tile_n.as_ref())? as u32,
            dimension(tile_k.as_ref())? as u32,
        )
        .map_err(error)?;
        create_workspace(shape, tile, None, MatmulAccumulation::Sequential).await
    }

    #[wasm_bindgen(js_name = createWithKernel)]
    pub async fn create_with_kernel(
        rows: Number,
        inner: Number,
        cols: Number,
        tile_m: Number,
        tile_n: Number,
        tile_k: Number,
        kernel: JsString,
    ) -> Result<WasmWgpuMatmul, JsValue> {
        let shape = MatmulShape::new(
            dimension(rows.as_ref())?,
            dimension(inner.as_ref())?,
            dimension(cols.as_ref())?,
        )
        .map_err(error)?;
        let tile = MatmulTile::new(
            dimension(tile_m.as_ref())? as u32,
            dimension(tile_n.as_ref())? as u32,
            dimension(tile_k.as_ref())? as u32,
        )
        .map_err(error)?;
        let kernel = kernel
            .as_string()
            .ok_or_else(|| error("kernel must be a string"))?
            .parse::<MatmulKernel>()
            .map_err(error)?;
        create_workspace(shape, tile, Some(kernel), MatmulAccumulation::Sequential).await
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "JS ABI exposes original numeric axes and independent kernel/accumulation choices"
    )]
    #[wasm_bindgen(js_name = createWithOptions)]
    pub async fn create_with_options(
        rows: Number,
        inner: Number,
        cols: Number,
        tile_m: Number,
        tile_n: Number,
        tile_k: Number,
        kernel: JsString,
        accumulation: JsString,
    ) -> Result<WasmWgpuMatmul, JsValue> {
        let shape = MatmulShape::new(
            dimension(rows.as_ref())?,
            dimension(inner.as_ref())?,
            dimension(cols.as_ref())?,
        )
        .map_err(error)?;
        let tile = MatmulTile::new(
            dimension(tile_m.as_ref())? as u32,
            dimension(tile_n.as_ref())? as u32,
            dimension(tile_k.as_ref())? as u32,
        )
        .map_err(error)?;
        let kernel = kernel
            .as_string()
            .ok_or_else(|| error("kernel must be a string"))?
            .parse::<MatmulKernel>()
            .map_err(error)?;
        let accumulation = accumulation
            .as_string()
            .ok_or_else(|| error("accumulation must be a string"))?
            .parse::<MatmulAccumulation>()
            .map_err(error)?;
        create_workspace(shape, tile, Some(kernel), accumulation).await
    }

    #[wasm_bindgen(js_name = tileMNK)]
    pub fn tile_mnk(&self) -> Vec<u32> {
        self.inner.tile().dimensions().to_vec()
    }

    #[wasm_bindgen(js_name = workgroupSize)]
    pub fn workgroup_size(&self) -> Vec<u32> {
        self.inner.workgroup_size().to_vec()
    }

    #[wasm_bindgen(js_name = outputsPerThread)]
    pub fn outputs_per_thread(&self) -> Vec<u32> {
        self.inner.outputs_per_thread().to_vec()
    }

    #[wasm_bindgen(getter)]
    pub fn kernel(&self) -> String {
        self.inner.kernel().as_str().to_owned()
    }

    #[wasm_bindgen(getter)]
    pub fn accumulation(&self) -> String {
        self.inner.accumulation().as_str().to_owned()
    }

    #[wasm_bindgen(js_name = shape)]
    pub fn shape(&self) -> Vec<u32> {
        let (m, k, n) = self.inner.shape().dimensions();
        vec![m as u32, k as u32, n as u32]
    }

    #[wasm_bindgen(getter, js_name = generation)]
    pub fn generation(&self) -> u64 {
        self.inner.generation()
    }

    #[wasm_bindgen(getter, js_name = outputIsCurrent)]
    pub fn output_is_current(&self) -> bool {
        self.inner.output_is_current()
    }

    #[wasm_bindgen(js_name = adapterInfo)]
    pub fn adapter_info(&self) -> Result<JsValue, JsValue> {
        let info = self.inner.adapter_info();
        serde_wasm_bindgen::to_value(&serde_json::json!({
            "name": info.name, "vendor": info.vendor, "device": info.device,
            "backend": format!("{:?}", info.backend), "device_type": format!("{:?}", info.device_type),
            "driver": info.driver, "driver_info": info.driver_info,
        })).map_err(error)
    }

    #[wasm_bindgen(js_name = upload)]
    pub fn upload(&mut self, lhs: Float32Array, rhs: Float32Array) -> Result<(), JsValue> {
        let lhs = float32_array(lhs.into())?;
        let rhs = float32_array(rhs.into())?;
        let (m, k, n) = self.inner.shape().dimensions();
        if lhs.length() as usize != m * k || rhs.length() as usize != k * n {
            return Err(error("operand lengths must match the workspace"));
        }
        self.inner
            .upload(&lhs.to_vec(), &rhs.to_vec())
            .map_err(error)
    }

    #[wasm_bindgen(js_name = uploadRhs)]
    pub fn upload_rhs(&mut self, rhs: Float32Array) -> Result<(), JsValue> {
        let rhs = float32_array(rhs.into())?;
        let (_, k, n) = self.inner.shape().dimensions();
        if rhs.length() as usize != k * n {
            return Err(error("RHS length must match the workspace"));
        }
        self.inner.upload_rhs(&rhs.to_vec()).map_err(error)
    }

    #[wasm_bindgen(js_name = setLhsFrom)]
    pub fn set_lhs_from(&mut self, source: &WasmWgpuMatmul) -> Result<(), JsValue> {
        self.inner.set_lhs_from(&source.inner).map_err(error)
    }

    /// Repetitions are validated before any JS-to-u32 narrowing.
    #[wasm_bindgen(js_name = dispatch)]
    pub fn dispatch(&mut self, repetitions: Option<Number>) -> Result<u64, JsValue> {
        let repetitions = match repetitions {
            None => 1,
            Some(value) => dimension(value.as_ref())? as u32,
        };
        self.inner.dispatch(repetitions).map_err(error)
    }

    #[wasm_bindgen(js_name = synchronize, unchecked_return_type = "Promise<void>")]
    pub fn synchronize(&self) -> Result<Promise, JsValue> {
        let completion = self.inner.synchronize_async().map_err(error)?;
        Ok(future_to_promise(async move {
            completion.await.map_err(error)?;
            Ok(JsValue::UNDEFINED)
        }))
    }

    /// Snapshot synchronously, then await mapping without holding a JS object borrow.
    #[wasm_bindgen(js_name = readback, unchecked_return_type = "Promise<Float32Array>")]
    pub fn readback(&self) -> Result<Promise, JsValue> {
        let snapshot = self.inner.snapshot().map_err(error)?;
        Ok(future_to_promise(async move {
            let values = snapshot.read_async().await.map_err(error)?;
            Ok(Float32Array::from(values.as_slice()).into())
        }))
    }
}

/// Exact rank workspace sharing Rust kernels, with no WASM CPU fallback.
#[wasm_bindgen(js_name = WgpuRank)]
pub struct WasmWgpuRank {
    inner: ResidentRank,
}

#[wasm_bindgen(js_class = WgpuRank)]
impl WasmWgpuRank {
    /// Construct from Rust-owned candidate geometry, without rebuilding the plan in JS.
    #[wasm_bindgen(js_name = createFromAdaptation)]
    pub async fn create_from_adaptation(
        session: &crate::rank_adaptation::WasmRankAdaptationSession,
        candidate_index: Number,
        timestamp_queries: Option<js_sys::Boolean>,
    ) -> Result<WasmWgpuRank, JsValue> {
        let timestamps = timestamp_request(timestamp_queries)?;
        let index = crate::utils::js_u32(candidate_index.as_ref(), "candidate index")?;
        let spec = session
            .wgpu_resident_candidate(index as usize)
            .map_err(error)?;
        let plan = Plan::try_new(spec.kind, spec.rows, spec.cols, spec.k, spec.tile_cols)
            .map_err(error)?;
        Ok(Self {
            inner: ResidentRank::new_async(rank_runtime(timestamps).await?, plan)
                .await
                .map_err(error)?,
        })
    }

    #[wasm_bindgen(js_name = create)]
    pub async fn create(
        kind: JsString,
        rows: Number,
        cols: Number,
        k: Number,
        tile_cols: Option<Number>,
        timestamp_queries: Option<js_sys::Boolean>,
    ) -> Result<WasmWgpuRank, JsValue> {
        let timestamps = timestamp_request(timestamp_queries)?;
        let kind = match kind.as_string().as_deref() {
            Some("topk") => Kind::TopK,
            Some("midk") => Kind::MidK,
            Some("bottomk") => Kind::BottomK,
            _ => return Err(error("kind must be topk, midk, or bottomk")),
        };
        let tile = tile_cols
            .as_ref()
            .map(|v| dimension(v.as_ref()))
            .transpose()?
            .unwrap_or(256);
        let plan = Plan::try_new(
            kind,
            dimension(rows.as_ref())? as u32,
            dimension(cols.as_ref())? as u32,
            dimension(k.as_ref())? as u32,
            tile as u32,
        )
        .map_err(error)?;
        Ok(Self {
            inner: ResidentRank::new_async(rank_runtime(timestamps).await?, plan)
                .await
                .map_err(error)?,
        })
    }

    #[wasm_bindgen(js_name = shape)]
    pub fn shape(&self) -> Vec<u32> {
        let p = self.inner.plan();
        vec![p.rows(), p.cols(), p.k()]
    }
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> String {
        self.inner.plan().kind().as_str().into()
    }
    #[wasm_bindgen(getter, js_name = tileCols)]
    pub fn tile_cols(&self) -> u32 {
        self.inner.plan().tile_cols()
    }
    #[wasm_bindgen(getter)]
    pub fn generation(&self) -> u64 {
        self.inner.generation()
    }
    #[wasm_bindgen(getter, js_name = outputIsCurrent)]
    pub fn output_is_current(&self) -> bool {
        self.inner.output_is_current()
    }

    #[wasm_bindgen(getter, js_name = timestampQueriesEnabled)]
    pub fn timestamp_queries_enabled(&self) -> bool {
        self.inner.timestamp_queries_enabled()
    }

    /// The promise owns query storage; later uploads or freeing the workspace are safe.
    #[wasm_bindgen(unchecked_return_type = "Promise<Record<string, unknown>>")]
    pub fn profile(&mut self, repetitions: Option<Number>) -> Result<Promise, JsValue> {
        let reps = repetitions
            .as_ref()
            .map(|v| dimension(v.as_ref()))
            .transpose()?
            .unwrap_or(1);
        let pending = self.inner.dispatch_profiled(reps as u32).map_err(error)?;
        Ok(future_to_promise(async move {
            let result = pending.read_async().await.map_err(error)?;
            crate::utils::json_to_js_value(&result.report().to_string())
        }))
    }

    #[wasm_bindgen(js_name = adapterInfo)]
    pub fn adapter_info(&self) -> Result<JsValue, JsValue> {
        let info = self.inner.adapter_info();
        crate::utils::json_to_js_value(
            &serde_json::json!({
                "name": info.name, "backend": format!("{:?}", info.backend),
                "device_type": format!("{:?}", info.device_type),
            })
            .to_string(),
        )
    }

    pub fn upload(&mut self, input: Float32Array) -> Result<(), JsValue> {
        let input = float32_array(input.into())?;
        if input.length() != self.inner.plan().input_elements() {
            return Err(error("input length must match the rank workspace"));
        }
        self.inner.upload(&input.to_vec()).map_err(error)
    }

    #[wasm_bindgen(js_name = setInputFromMatmul)]
    pub fn set_input_from_matmul(&mut self, source: &WasmWgpuMatmul) -> Result<(), JsValue> {
        self.inner
            .set_input_from_matmul(&source.inner)
            .map_err(error)
    }

    #[wasm_bindgen(js_name = dispatchFromMatmul)]
    pub fn dispatch_from_matmul(
        &mut self,
        source: &mut WasmWgpuMatmul,
        repetitions: Option<Number>,
    ) -> Result<u64, JsValue> {
        let reps = repetitions
            .as_ref()
            .map(|v| dimension(v.as_ref()))
            .transpose()?
            .unwrap_or(1);
        self.inner
            .dispatch_from_matmul(&mut source.inner, reps as u32)
            .map_err(error)
    }

    pub fn dispatch(&mut self, repetitions: Option<Number>) -> Result<u64, JsValue> {
        let reps = repetitions
            .as_ref()
            .map(|v| dimension(v.as_ref()))
            .transpose()?
            .unwrap_or(1);
        self.inner.dispatch(reps as u32).map_err(error)
    }

    #[wasm_bindgen(js_name = readback, unchecked_return_type = "Promise<{values: Float32Array; indices: Int32Array; generation: bigint}>")]
    pub fn readback(&self) -> Result<Promise, JsValue> {
        let snapshot = self.inner.snapshot().map_err(error)?;
        let generation = snapshot.generation();
        Ok(future_to_promise(async move {
            let output = snapshot.read_async().await.map_err(error)?;
            let result = js_sys::Object::new();
            js_sys::Reflect::set(
                &result,
                &"values".into(),
                &Float32Array::from(output.values.as_slice()),
            )?;
            js_sys::Reflect::set(
                &result,
                &"indices".into(),
                &Int32Array::from(output.indices.as_slice()),
            )?;
            js_sys::Reflect::set(
                &result,
                &"generation".into(),
                &js_sys::BigInt::from(generation),
            )?;
            Ok(result.into())
        }))
    }

    #[wasm_bindgen(js_name = synchronize, unchecked_return_type = "Promise<void>")]
    pub fn synchronize(&self) -> Result<Promise, JsValue> {
        let completion = self.inner.synchronize_async().map_err(error)?;
        Ok(future_to_promise(async move {
            completion.await.map_err(error)?;
            Ok(JsValue::UNDEFINED)
        }))
    }
}
