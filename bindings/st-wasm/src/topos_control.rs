use serde::Deserialize;
use serde_json::Value;
use st_tensor::{
    TensorError, ToposControlPlanOptions, ToposControlSignal, ToposControlSignalInput,
    ToposInferenceHintsInput, ToposTrainingHintsInput, ToposZSpaceProjectionOptions,
};

const TOPOS_CONTROL_SIGNAL_INPUT_KEYS: &[&str] = &[
    "curvature",
    "tolerance",
    "saturation",
    "porosity",
    "max_depth",
    "max_volume",
    "observed_depth",
    "visited_volume",
];

const TOPOS_OPTIMIZER_SNAPSHOT_INPUT_KEYS: &[&str] = &[
    "signal",
    "sequence",
    "hyper_learning_rate",
    "real_learning_rate",
    "options",
    "training_hints",
    "inference_hints",
];

const TOPOS_CONTROL_OPTIONS_INPUT_KEYS: &[&str] = &["training_gain", "inference"];
const TOPOS_INFERENCE_OPTIONS_INPUT_KEYS: &[&str] = &[
    "gain",
    "base_temperature",
    "base_top_p",
    "min_temperature",
    "max_temperature",
    "min_top_p",
    "max_top_p",
    "base_frequency_penalty",
    "base_presence_penalty",
];
const TOPOS_TRAINING_HINTS_INPUT_KEYS: &[&str] = &[
    "learning_rate_scale",
    "regularization_scale",
    "step_damping",
    "gradient_bias_scale",
    "clip_scale",
    "momentum_damping",
    "vector",
];
const TOPOS_INFERENCE_HINTS_INPUT_KEYS: &[&str] = &[
    "temperature_scale",
    "top_p_scale",
    "sampling_focus",
    "frequency_penalty_bias",
    "presence_penalty_bias",
    "context_weight",
    "vector",
];

#[derive(Clone, Copy, Debug)]
pub struct ToposOptimizerSnapshotRequest {
    pub signal: ToposControlSignalInput,
    pub sequence: u64,
    pub hyper_learning_rate: f32,
    pub real_learning_rate: f32,
    pub options: ToposControlPlanOptions,
    pub training_hints: Option<ToposTrainingHintsInput>,
    pub inference_hints: Option<ToposInferenceHintsInput>,
}

#[derive(Deserialize)]
struct ToposOptimizerSnapshotRequestWire {
    signal: Value,
    sequence: u64,
    hyper_learning_rate: f32,
    real_learning_rate: f32,
    #[serde(default)]
    options: Option<Value>,
    #[serde(default)]
    training_hints: Option<Value>,
    #[serde(default)]
    inference_hints: Option<Value>,
}

fn validate_value_keys(value: &Value, label: &str, allowed: &[&str]) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(format!("unsupported {label} key: {key}"));
    }
    Ok(())
}

fn topos_control_signal_input_from_value(value: Value) -> Result<ToposControlSignalInput, String> {
    validate_value_keys(
        &value,
        "Topos control signal",
        TOPOS_CONTROL_SIGNAL_INPUT_KEYS,
    )?;
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn topos_control_signal_input_from_json(
    input_json: &str,
) -> Result<ToposControlSignalInput, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    topos_control_signal_input_from_value(value)
}

fn topos_optimizer_snapshot_request_from_value(
    value: Value,
) -> Result<ToposOptimizerSnapshotRequest, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "Topos optimizer snapshot input must be an object".to_owned())?;
    if let Some(key) = object
        .keys()
        .find(|key| !TOPOS_OPTIMIZER_SNAPSHOT_INPUT_KEYS.contains(&key.as_str()))
    {
        return Err(format!("unsupported Topos optimizer snapshot key: {key}"));
    }
    let wire: ToposOptimizerSnapshotRequestWire =
        serde_json::from_value(value).map_err(|error| error.to_string())?;
    let options = match wire.options {
        Some(options) => {
            validate_value_keys(
                &options,
                "Topos control options",
                TOPOS_CONTROL_OPTIONS_INPUT_KEYS,
            )?;
            if let Some(inference) = options.get("inference") {
                validate_value_keys(
                    inference,
                    "Topos inference options",
                    TOPOS_INFERENCE_OPTIONS_INPUT_KEYS,
                )?;
            }
            serde_json::from_value(options).map_err(|error| error.to_string())?
        }
        None => ToposControlPlanOptions::default(),
    };
    let training_hints = match wire.training_hints {
        Some(hints) => {
            validate_value_keys(
                &hints,
                "Topos training hints",
                TOPOS_TRAINING_HINTS_INPUT_KEYS,
            )?;
            Some(serde_json::from_value(hints).map_err(|error| error.to_string())?)
        }
        None => None,
    };
    let inference_hints = match wire.inference_hints {
        Some(hints) => {
            validate_value_keys(
                &hints,
                "Topos inference hints",
                TOPOS_INFERENCE_HINTS_INPUT_KEYS,
            )?;
            Some(serde_json::from_value(hints).map_err(|error| error.to_string())?)
        }
        None => None,
    };
    Ok(ToposOptimizerSnapshotRequest {
        signal: topos_control_signal_input_from_value(wire.signal)?,
        sequence: wire.sequence,
        hyper_learning_rate: wire.hyper_learning_rate,
        real_learning_rate: wire.real_learning_rate,
        options,
        training_hints,
        inference_hints,
    })
}

fn topos_optimizer_snapshot_request_from_json(
    input_json: &str,
) -> Result<ToposOptimizerSnapshotRequest, String> {
    let value = serde_json::from_str(input_json).map_err(|error| error.to_string())?;
    topos_optimizer_snapshot_request_from_value(value)
}

#[cfg(test)]
use st_tensor::{
    TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION, TOPOS_OPTIMIZER_SNAPSHOT_CONTRACT_VERSION,
    TOPOS_OPTIMIZER_SNAPSHOT_MAX_SEQUENCE, TOPOS_ZSPACE_PROJECTION_CONTRACT_VERSION,
    TOPOS_ZSPACE_PROJECTION_MAX_GRADIENT_DIM,
};

#[cfg(target_arch = "wasm32")]
use serde::Serialize;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::utils::js_error;

#[cfg(target_arch = "wasm32")]
fn to_json_compatible_js(value: &Value) -> Result<JsValue, JsValue> {
    value
        .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
        .map_err(js_error)
}

/// Build the shared Topos control-signal payload without introducing WASM semantics.
pub fn topos_control_signal_value(input: ToposControlSignalInput) -> Result<Value, TensorError> {
    let signal = ToposControlSignal::from_input(input)?;
    let mut payload =
        serde_json::to_value(signal.payload()).expect("Topos control payload is serializable");
    let payload_map = payload
        .as_object_mut()
        .expect("Topos control payload is an object");
    payload_map.insert(
        "execution_client".to_owned(),
        Value::String("wasm".to_owned()),
    );
    if let Some(route) = payload_map
        .get_mut("runtime_route")
        .and_then(Value::as_object_mut)
    {
        route.insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
    }
    Ok(payload)
}

/// Build one optimizer application snapshot through the shared Rust contract.
pub fn topos_optimizer_snapshot_value(
    request: ToposOptimizerSnapshotRequest,
) -> Result<Value, TensorError> {
    let signal = ToposControlSignal::from_input(request.signal)?;
    let snapshot = signal.optimizer_snapshot(
        request.sequence,
        request.hyper_learning_rate,
        request.real_learning_rate,
        request.options,
        request.training_hints,
        request.inference_hints,
    )?;
    let mut payload =
        serde_json::to_value(snapshot).expect("Topos optimizer snapshot payload is serializable");
    payload
        .as_object_mut()
        .expect("Topos optimizer snapshot payload is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
    Ok(payload)
}

/// Project a Topos control signal through the shared Rust Z-space contract.
pub fn topos_zspace_projection_value(
    input: ToposControlSignalInput,
    gradient_dim: usize,
) -> Result<Value, TensorError> {
    let signal = ToposControlSignal::from_input(input)?;
    let projection = signal.zspace_projection(ToposZSpaceProjectionOptions { gradient_dim })?;
    let mut payload = serde_json::to_value(projection.payload())
        .expect("Topos Z-space projection payload is serializable");
    payload
        .as_object_mut()
        .expect("Topos Z-space projection payload is an object")
        .insert(
            "execution_client".to_owned(),
            Value::String("wasm".to_owned()),
        );
    Ok(payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposControlSignalJson)]
pub fn topos_control_signal_json(input_json: &str) -> Result<String, JsValue> {
    let input = topos_control_signal_input_from_json(input_json).map_err(js_error)?;
    let payload = topos_control_signal_value(input).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposControlSignalObject)]
pub fn topos_control_signal_object(input: &JsValue) -> Result<JsValue, JsValue> {
    let input = serde_wasm_bindgen::from_value::<Value>(input.clone()).map_err(js_error)?;
    let input = topos_control_signal_input_from_value(input).map_err(js_error)?;
    let payload = topos_control_signal_value(input).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposOptimizerSnapshotJson)]
pub fn topos_optimizer_snapshot_json(input_json: &str) -> Result<String, JsValue> {
    let request = topos_optimizer_snapshot_request_from_json(input_json).map_err(js_error)?;
    let payload = topos_optimizer_snapshot_value(request).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposOptimizerSnapshotObject)]
pub fn topos_optimizer_snapshot_object(input: &JsValue) -> Result<JsValue, JsValue> {
    let input = serde_wasm_bindgen::from_value::<Value>(input.clone()).map_err(js_error)?;
    let request = topos_optimizer_snapshot_request_from_value(input).map_err(js_error)?;
    let payload = topos_optimizer_snapshot_value(request).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposZSpaceProjectionJson)]
pub fn topos_zspace_projection_json(
    input_json: &str,
    gradient_dim: usize,
) -> Result<String, JsValue> {
    let input = topos_control_signal_input_from_json(input_json).map_err(js_error)?;
    let payload = topos_zspace_projection_value(input, gradient_dim).map_err(js_error)?;
    serde_json::to_string(&payload).map_err(js_error)
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = toposZSpaceProjectionObject)]
pub fn topos_zspace_projection_object(
    input: &JsValue,
    gradient_dim: usize,
) -> Result<JsValue, JsValue> {
    let input = serde_wasm_bindgen::from_value::<Value>(input.clone()).map_err(js_error)?;
    let input = topos_control_signal_input_from_value(input).map_err(js_error)?;
    let payload = topos_zspace_projection_value(input, gradient_dim).map_err(js_error)?;
    to_json_compatible_js(&payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_control_signal_uses_st_tensor_contract() {
        let payload = topos_control_signal_value(ToposControlSignalInput {
            max_depth: 10,
            max_volume: 100,
            observed_depth: 4,
            visited_volume: 25,
            ..ToposControlSignalInput::default()
        })
        .expect("valid control signal");

        assert_eq!(payload["kind"], "spiraltorch.topos_control_signal");
        assert_eq!(
            payload["contract_version"],
            TOPOS_CONTROL_SIGNAL_CONTRACT_VERSION
        );
        assert_eq!(payload["semantic_backend"], "rust");
        assert_eq!(payload["execution_client"], "wasm");
        let closure_pressure = payload["closure_pressure"]
            .as_f64()
            .expect("closure pressure is numeric");
        assert!((closure_pressure - 0.4).abs() < 1e-6);
        assert_eq!(payload["runtime_route"]["semantic_backend"], "rust");
    }

    #[test]
    fn wasm_control_signal_matches_rust_payload_except_client_metadata() {
        let input = ToposControlSignalInput {
            porosity: 0.35,
            max_depth: 12,
            max_volume: 80,
            observed_depth: 7,
            visited_volume: 31,
            ..ToposControlSignalInput::default()
        };
        let signal = ToposControlSignal::from_input(input).expect("valid control signal");
        let expected = serde_json::to_value(signal.payload()).expect("serializable Rust payload");
        let mut actual = topos_control_signal_value(input).expect("valid WASM payload");
        let actual_map = actual.as_object_mut().expect("WASM payload object");
        actual_map.remove("execution_client");
        actual_map
            .get_mut("runtime_route")
            .and_then(Value::as_object_mut)
            .expect("runtime route object")
            .remove("execution_client");

        assert_eq!(actual, expected);
    }

    #[test]
    fn wasm_control_signal_rejects_invalid_topology() {
        let result = topos_control_signal_value(ToposControlSignalInput {
            curvature: f32::NAN,
            ..ToposControlSignalInput::default()
        });
        assert!(matches!(result, Err(TensorError::NonFiniteValue { .. })));
    }

    #[test]
    fn wasm_control_signal_ingress_rejects_unknown_keys() {
        let error = topos_control_signal_input_from_json(r#"{"porosity":0.25,"porostiy":0.75}"#)
            .expect_err("unknown Topos control key must fail closed");

        assert!(error.contains("unsupported Topos control signal key: porostiy"));
    }

    #[test]
    fn wasm_optimizer_snapshot_matches_rust_except_client_metadata() {
        let request = ToposOptimizerSnapshotRequest {
            signal: ToposControlSignalInput {
                porosity: 0.25,
                max_depth: 10,
                max_volume: 100,
                observed_depth: 4,
                visited_volume: 25,
                ..ToposControlSignalInput::default()
            },
            sequence: 7,
            hyper_learning_rate: 0.04,
            real_learning_rate: 0.02,
            options: ToposControlPlanOptions {
                training_gain: 0.5,
                ..ToposControlPlanOptions::default()
            },
            training_hints: Some(ToposTrainingHintsInput {
                learning_rate_scale: Some(0.7),
                clip_scale: Some(0.8),
                ..ToposTrainingHintsInput::default()
            }),
            inference_hints: None,
        };
        let signal = ToposControlSignal::from_input(request.signal).expect("valid signal");
        let expected = serde_json::to_value(
            signal
                .optimizer_snapshot(
                    request.sequence,
                    request.hyper_learning_rate,
                    request.real_learning_rate,
                    request.options,
                    request.training_hints,
                    request.inference_hints,
                )
                .expect("valid Rust snapshot"),
        )
        .expect("serializable Rust snapshot");
        let mut actual =
            topos_optimizer_snapshot_value(request).expect("valid WASM snapshot payload");
        actual
            .as_object_mut()
            .expect("WASM snapshot object")
            .remove("execution_client");

        assert_eq!(actual, expected);
        assert_eq!(
            actual["contract_version"],
            TOPOS_OPTIMIZER_SNAPSHOT_CONTRACT_VERSION
        );
        assert_eq!(actual["sequence"], 7);
        assert_eq!(
            actual["optimizer_application"]["rate_scale"],
            actual["control"]["training_plan"]["rate_scale"]
        );
        assert_eq!(
            actual["optimizer_application"]["scope"],
            "learning_rate_and_gradient_state"
        );
        assert_eq!(
            actual["optimizer_application"]["effective_gradient_bias_scale"],
            actual["control"]["training_plan"]["effective_gradient_bias_scale"]
        );
        assert_eq!(
            actual["optimizer_application"]["effective_gradient_clip_scale"],
            actual["control"]["training_plan"]["effective_gradient_clip_scale"]
        );
        assert_eq!(
            actual["optimizer_application"]["effective_momentum_damping"],
            actual["control"]["training_plan"]["effective_momentum_damping"]
        );
        assert_eq!(
            actual["optimizer_application"]["gradient_bias_normalization"],
            "raw_gradient_rms"
        );
        assert_eq!(
            actual["optimizer_application"]["gradient_clip_normalization"],
            "biased_gradient_rms"
        );
        assert_eq!(
            actual["optimizer_application"]["gradient_bias_basis"]
                .as_array()
                .expect("WASM optimizer bias basis")
                .len(),
            10
        );
    }

    #[test]
    fn wasm_optimizer_snapshot_ingress_and_numeric_boundaries_fail_closed() {
        let unknown = topos_optimizer_snapshot_request_from_json(
            r#"{"signal":{},"sequence":1,"hyper_learning_rate":0.04,"real_learning_rate":0.02,"commander":"python"}"#,
        )
        .expect_err("unknown snapshot key must fail closed");
        assert!(unknown.contains("unsupported Topos optimizer snapshot key: commander"));

        let unknown_hint = topos_optimizer_snapshot_request_from_json(
            r#"{"signal":{},"sequence":1,"hyper_learning_rate":0.04,"real_learning_rate":0.02,"training_hints":{"clip_sclae":0.5}}"#,
        )
        .expect_err("unknown training hint must fail closed");
        assert!(unknown_hint.contains("unsupported Topos training hints key: clip_sclae"));

        let unknown_option = topos_optimizer_snapshot_request_from_json(
            r#"{"signal":{},"sequence":1,"hyper_learning_rate":0.04,"real_learning_rate":0.02,"options":{"inference":{"base_temperatur":0.8}}}"#,
        )
        .expect_err("unknown inference option must fail closed");
        assert!(unknown_option.contains("unsupported Topos inference options key: base_temperatur"));

        let unsafe_sequence = topos_optimizer_snapshot_value(ToposOptimizerSnapshotRequest {
            signal: ToposControlSignalInput::default(),
            sequence: TOPOS_OPTIMIZER_SNAPSHOT_MAX_SEQUENCE + 1,
            hyper_learning_rate: 0.04,
            real_learning_rate: 0.02,
            options: ToposControlPlanOptions::default(),
            training_hints: None,
            inference_hints: None,
        })
        .expect_err("unsafe sequence must fail closed");
        assert!(matches!(unsafe_sequence, TensorError::InvalidValue { .. }));

        let zero_rate = topos_optimizer_snapshot_value(ToposOptimizerSnapshotRequest {
            signal: ToposControlSignalInput::default(),
            sequence: 1,
            hyper_learning_rate: 0.0,
            real_learning_rate: 0.02,
            options: ToposControlPlanOptions::default(),
            training_hints: None,
            inference_hints: None,
        })
        .expect_err("zero learning rate must fail closed");
        assert!(matches!(
            zero_rate,
            TensorError::NonPositiveLearningRate { .. }
        ));
    }

    #[test]
    fn wasm_zspace_projection_matches_rust_except_client_metadata() {
        let input = ToposControlSignalInput {
            porosity: 0.25,
            max_depth: 10,
            max_volume: 100,
            observed_depth: 4,
            visited_volume: 25,
            ..ToposControlSignalInput::default()
        };
        let signal = ToposControlSignal::from_input(input).expect("valid control signal");
        let expected = serde_json::to_value(
            signal
                .zspace_projection(ToposZSpaceProjectionOptions { gradient_dim: 8 })
                .expect("valid projection")
                .payload(),
        )
        .expect("serializable Rust projection");
        let mut actual = topos_zspace_projection_value(input, 8).expect("valid WASM projection");
        actual
            .as_object_mut()
            .expect("WASM projection object")
            .remove("execution_client");

        assert_eq!(actual, expected);
        assert_eq!(
            expected["contract_version"],
            TOPOS_ZSPACE_PROJECTION_CONTRACT_VERSION
        );
        assert_eq!(expected["semantic_backend"], "rust");
        assert_eq!(expected["gradient_dim"], 8);
        assert_eq!(expected["gradient"].as_array().map(Vec::len), Some(8));
    }

    #[test]
    fn wasm_zspace_projection_rejects_unsafe_dimensions() {
        for gradient_dim in [0, TOPOS_ZSPACE_PROJECTION_MAX_GRADIENT_DIM + 1] {
            let error =
                topos_zspace_projection_value(ToposControlSignalInput::default(), gradient_dim)
                    .expect_err("unsafe projection dimension must fail closed");
            assert!(matches!(error, TensorError::InvalidValue { .. }));
        }
    }
}
