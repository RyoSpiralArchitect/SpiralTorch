use st_tensor::{
    AutogradBackwardReport, AutogradTensor as RustAutogradTensor, Tensor,
    AUTOGRAD_CONTRACT_VERSION, AUTOGRAD_SEMANTIC_OWNER,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use serde_json::Value;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

#[derive(Clone)]
struct AutogradTransport {
    inner: RustAutogradTensor,
}

impl AutogradTransport {
    fn new(
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        requires_grad: bool,
    ) -> Result<Self, String> {
        let value = Tensor::from_vec(rows, cols, values).map_err(|error| error.to_string())?;
        let inner = RustAutogradTensor::from_tensor(value, requires_grad)
            .map_err(|error| error.to_string())?;
        Ok(Self { inner })
    }

    fn from_inner(inner: RustAutogradTensor) -> Self {
        Self { inner }
    }

    fn backward(&self) -> Result<AutogradBackwardReport, String> {
        self.inner.backward().map_err(|error| error.to_string())
    }

    fn backward_with_values(&self, values: Vec<f32>) -> Result<AutogradBackwardReport, String> {
        let (rows, cols) = self.inner.shape();
        let seed = Tensor::from_vec(rows, cols, values).map_err(|error| error.to_string())?;
        self.inner
            .backward_with_grad(&seed)
            .map_err(|error| error.to_string())
    }

    fn vector_jacobian_product(&self, input: &Self, values: Vec<f32>) -> Result<Tensor, String> {
        let (rows, cols) = self.inner.shape();
        let seed = Tensor::from_vec(rows, cols, values).map_err(|error| error.to_string())?;
        self.inner
            .vector_jacobian_product(&input.inner, &seed)
            .map_err(|error| error.to_string())
    }
}

/// Returns the native semantic contract version exposed by the browser client.
#[cfg_attr(
    target_arch = "wasm32",
    wasm_bindgen(js_name = autogradContractVersion)
)]
pub fn autograd_contract_version() -> String {
    AUTOGRAD_CONTRACT_VERSION.to_owned()
}

/// Returns the crate that owns reverse-mode semantics.
#[cfg_attr(
    target_arch = "wasm32",
    wasm_bindgen(js_name = autogradSemanticOwner)
)]
pub fn autograd_semantic_owner() -> String {
    AUTOGRAD_SEMANTIC_OWNER.to_owned()
}

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Browser handle over the same immutable graph used by native Rust and Python.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct AutogradTensor {
    transport: AutogradTransport,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl AutogradTensor {
    #[wasm_bindgen(constructor)]
    pub fn new(
        rows: usize,
        cols: usize,
        values: Vec<f32>,
        requires_grad: bool,
    ) -> Result<AutogradTensor, JsValue> {
        AutogradTransport::new(rows, cols, values, requires_grad)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = id)]
    pub fn id_string(&self) -> String {
        self.transport.inner.id().to_string()
    }

    #[wasm_bindgen(js_name = rows)]
    pub fn rows(&self) -> usize {
        self.transport.inner.shape().0
    }

    #[wasm_bindgen(js_name = cols)]
    pub fn cols(&self) -> usize {
        self.transport.inner.shape().1
    }

    #[wasm_bindgen(js_name = requiresGrad)]
    pub fn requires_grad(&self) -> bool {
        self.transport.inner.requires_grad()
    }

    #[wasm_bindgen(js_name = operationName)]
    pub fn operation_name(&self) -> String {
        self.transport.inner.operation_name().to_owned()
    }

    #[wasm_bindgen(js_name = values)]
    pub fn values(&self) -> Vec<f32> {
        self.transport.inner.value().data().to_vec()
    }

    #[wasm_bindgen(js_name = hasGradient)]
    pub fn has_gradient(&self) -> bool {
        self.transport.inner.grad().is_some()
    }

    #[wasm_bindgen(js_name = gradientValues)]
    pub fn gradient_values(&self) -> Result<Vec<f32>, JsValue> {
        self.transport
            .inner
            .grad()
            .map(|gradient| gradient.data().to_vec())
            .ok_or_else(|| js_error("autograd gradient is not available"))
    }

    pub fn detach(&self) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .detach()
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = zeroGrad)]
    pub fn zero_grad(&self) {
        self.transport.inner.zero_grad();
    }

    #[wasm_bindgen(js_name = zeroGradGraph)]
    pub fn zero_grad_graph(&self) {
        self.transport.inner.zero_grad_graph();
    }

    pub fn add(&self, rhs: &AutogradTensor) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .add(&rhs.transport.inner)
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    pub fn sub(&self, rhs: &AutogradTensor) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .sub(&rhs.transport.inner)
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    pub fn hadamard(&self, rhs: &AutogradTensor) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .hadamard(&rhs.transport.inner)
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    pub fn matmul(&self, rhs: &AutogradTensor) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .matmul(&rhs.transport.inner)
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    pub fn scale(&self, factor: f32) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .scale(factor)
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    pub fn transpose(&self) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .transpose()
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    pub fn sum(&self) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .sum()
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    pub fn mean(&self) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .mean()
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    pub fn dot(&self, rhs: &AutogradTensor) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .dot(&rhs.transport.inner)
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = meanSquaredError)]
    pub fn mean_squared_error(&self, target: &AutogradTensor) -> Result<AutogradTensor, JsValue> {
        self.transport
            .inner
            .mean_squared_error(&target.transport.inner)
            .map(AutogradTransport::from_inner)
            .map(|transport| Self { transport })
            .map_err(js_error)
    }

    pub fn item(&self) -> Result<f32, JsValue> {
        self.transport.inner.item_f32().map_err(js_error)
    }

    pub fn backward(&self) -> Result<JsValue, JsValue> {
        let report = self.transport.backward().map_err(js_error)?;
        to_json_compatible_js(&report.contract_payload())
    }

    #[wasm_bindgen(js_name = backwardWithGrad)]
    pub fn backward_with_grad(&self, values: Vec<f32>) -> Result<JsValue, JsValue> {
        let report = self
            .transport
            .backward_with_values(values)
            .map_err(js_error)?;
        to_json_compatible_js(&report.contract_payload())
    }

    #[wasm_bindgen(js_name = vectorJacobianProduct)]
    pub fn vector_jacobian_product(
        &self,
        input: &AutogradTensor,
        values: Vec<f32>,
    ) -> Result<Vec<f32>, JsValue> {
        self.transport
            .vector_jacobian_product(&input.transport, values)
            .map(|gradient| gradient.data().to_vec())
            .map_err(js_error)
    }

    #[wasm_bindgen(js_name = graphSummary)]
    pub fn graph_summary(&self) -> Result<JsValue, JsValue> {
        to_json_compatible_js(&self.transport.inner.graph_summary().contract_payload())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_transport_delegates_branching_gradients_to_rust() {
        let x = AutogradTransport::new(1, 3, vec![1.0, 2.0, -1.0], true).unwrap();
        let squared = x.inner.hadamard(&x.inner).unwrap();
        let linear = x.inner.scale(3.0).unwrap();
        let loss = AutogradTransport::from_inner(squared.add(&linear).unwrap().sum().unwrap());

        let receipt = loss.backward().unwrap().contract_payload();

        assert_eq!(x.inner.grad().unwrap().data(), &[5.0, 7.0, 1.0]);
        assert_eq!(receipt["contract_version"], AUTOGRAD_CONTRACT_VERSION);
        assert_eq!(receipt["semantic_owner"], AUTOGRAD_SEMANTIC_OWNER);
        assert_eq!(receipt["leaf_gradient_count"], 1);
    }

    #[test]
    fn wasm_transport_requires_explicit_seed_for_vector_vjp() {
        let x = AutogradTransport::new(1, 2, vec![2.0, 3.0], true).unwrap();
        let y = AutogradTransport::from_inner(x.inner.scale(4.0).unwrap());

        let error = y.backward().unwrap_err();
        assert!(error.contains("explicit output gradient"));

        y.backward_with_values(vec![0.5, -1.0]).unwrap();
        assert_eq!(x.inner.grad().unwrap().data(), &[2.0, -4.0]);
    }

    #[test]
    fn wasm_transport_vjp_is_side_effect_free_and_disconnect_safe() {
        let x = AutogradTransport::new(1, 2, vec![2.0, -3.0], true).unwrap();
        let output = AutogradTransport::from_inner(x.inner.hadamard(&x.inner).unwrap());
        let disconnected = AutogradTransport::new(1, 2, vec![4.0, 5.0], true).unwrap();

        let gradient = output.vector_jacobian_product(&x, vec![0.5, -2.0]).unwrap();
        let disconnected_gradient = output
            .vector_jacobian_product(&disconnected, vec![1.0, 1.0])
            .unwrap();

        assert_eq!(gradient.data(), &[2.0, 12.0]);
        assert_eq!(disconnected_gradient.data(), &[0.0, 0.0]);
        assert!(x.inner.grad().is_none());
    }

    #[test]
    fn wasm_metadata_identifies_the_single_semantic_owner() {
        let x = AutogradTransport::new(1, 1, vec![2.0], true).unwrap();
        let summary = x.inner.graph_summary().contract_payload();

        assert_eq!(autograd_contract_version(), "spiraltorch.autograd.v1");
        assert_eq!(autograd_semantic_owner(), "st-tensor");
        assert_eq!(summary["contract_version"], AUTOGRAD_CONTRACT_VERSION);
        assert_eq!(summary["semantic_owner"], AUTOGRAD_SEMANTIC_OWNER);
    }
}
