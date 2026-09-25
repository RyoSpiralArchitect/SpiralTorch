//! Browser-facing geometry transforms backed by the same st-vision pipeline as native Rust.
use js_sys::Float32Array;
use st_backend_wgpu::transform::TransformDispatcher;
use st_vision::{
    CenterCrop, ImageTensor, RandomHorizontalFlip, Resize, TransformOperation, TransformPipeline,
};
use wasm_bindgen::prelude::*;

fn error(err: impl std::fmt::Display) -> JsValue {
    js_sys::Error::new(&err.to_string()).into()
}

#[wasm_bindgen(js_name = VisionTransformPipeline)]
pub struct WasmVisionTransformPipeline {
    inner: TransformPipeline,
    gpu: bool,
    adapter_info: Option<String>,
}

#[wasm_bindgen(js_class = VisionTransformPipeline)]
impl WasmVisionTransformPipeline {
    #[wasm_bindgen(js_name = createCpu)]
    pub fn create_cpu(seed: u32) -> Self {
        Self {
            inner: TransformPipeline::with_seed(u64::from(seed)),
            gpu: false,
            adapter_info: None,
        }
    }

    #[wasm_bindgen(js_name = createGpu)]
    pub async fn create_gpu(seed: u32) -> Result<Self, JsValue> {
        let runtime = crate::wgpu_resident::ensure_runtime().await?;
        let info = runtime.adapter_info();
        let adapter_info = serde_json::json!({
            "name": info.name,
            "backend": format!("{:?}", info.backend),
            "device_type": format!("{:?}", info.device_type),
        })
        .to_string();
        let dispatcher = TransformDispatcher::with_gpu(
            runtime.context().shared_device(),
            runtime.context().shared_queue(),
            "embedded-browser-shaders",
        )
        .map_err(error)?;
        Ok(Self {
            inner: TransformPipeline::with_seed(u64::from(seed)).with_gpu_dispatcher(dispatcher),
            gpu: true,
            adapter_info: Some(adapter_info),
        })
    }

    #[wasm_bindgen(js_name = addResize)]
    pub fn add_resize(&mut self, height: u32, width: u32) -> Result<(), JsValue> {
        self.inner.add(TransformOperation::Resize(
            Resize::new(height as usize, width as usize).map_err(error)?,
        ));
        Ok(())
    }

    #[wasm_bindgen(js_name = addCenterCrop)]
    pub fn add_center_crop(&mut self, height: u32, width: u32) -> Result<(), JsValue> {
        self.inner.add(TransformOperation::CenterCrop(
            CenterCrop::new(height as usize, width as usize).map_err(error)?,
        ));
        Ok(())
    }

    #[wasm_bindgen(js_name = addRandomHorizontalFlip)]
    pub fn add_random_horizontal_flip(&mut self, probability: f32) -> Result<(), JsValue> {
        self.inner.add(TransformOperation::RandomHorizontalFlip(
            RandomHorizontalFlip::new(probability).map_err(error)?,
        ));
        Ok(())
    }

    #[wasm_bindgen(js_name = apply)]
    pub async fn apply(
        &mut self,
        channels: u32,
        height: u32,
        width: u32,
        data: JsValue,
    ) -> Result<WasmVisionImage, JsValue> {
        let data = crate::wgpu_tensor::values(data)?.to_vec();
        let mut image = ImageTensor::new(channels as usize, height as usize, width as usize, data)
            .map_err(error)?;
        if self.gpu {
            self.inner
                .apply_geometry_async(&mut image)
                .await
                .map_err(error)?;
        } else {
            self.inner.apply(&mut image).map_err(error)?;
        }
        Ok(WasmVisionImage { inner: image })
    }

    #[wasm_bindgen(getter, js_name = backend)]
    pub fn backend(&self) -> String {
        if self.gpu { "webgpu" } else { "cpu" }.into()
    }

    #[wasm_bindgen(getter, js_name = adapterInfo)]
    pub fn adapter_info(&self) -> Option<String> {
        self.adapter_info.clone()
    }
}

#[wasm_bindgen(js_name = VisionImage)]
pub struct WasmVisionImage {
    inner: ImageTensor,
}

#[wasm_bindgen(js_class = VisionImage)]
impl WasmVisionImage {
    #[wasm_bindgen(getter)]
    pub fn shape(&self) -> Vec<u32> {
        let (channels, height, width) = self.inner.shape();
        vec![channels as u32, height as u32, width as u32]
    }

    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Float32Array {
        Float32Array::from(self.inner.as_slice())
    }
}
