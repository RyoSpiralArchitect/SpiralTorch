//! Generic acceptance handle over the existing Rust parameter transaction.
use super::*;
#[cfg(feature = "webgpu")]
use st_backend_wgpu::resident_training::parameters::ResidentParameterUpdate;

#[wasm_bindgen(js_name = ResidentParameterUpdate)]
pub struct WasmResidentParameterUpdate {
    #[cfg(feature = "webgpu")]
    pub(super) inner: ResidentParameterUpdate,
}

#[cfg(feature = "webgpu")]
#[wasm_bindgen(js_class = ResidentParameterUpdate)]
impl WasmResidentParameterUpdate {
    #[wasm_bindgen(getter, js_name = attemptedRevision)]
    pub fn attempted_revision(&self) -> u64 {
        self.inner.revision()
    }
    /// Explicit flag readback. A submitted update is not proof of acceptance.
    #[wasm_bindgen(unchecked_return_type = "Promise<bigint>")]
    pub fn read(&self) -> Result<Promise, JsValue> {
        let snapshot = self.inner.snapshot().map_err(js_error)?;
        Ok(future_to_promise(async move {
            Ok(js_sys::BigInt::from(snapshot.read_async().await.map_err(js_error)?).into())
        }))
    }
}
