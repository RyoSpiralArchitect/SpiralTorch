// SPDX-License-Identifier: AGPL-3.0-or-later

//! Replayable cross-client protocol for stochastic Schrodinger dynamics.
//!
//! The numerical equation and analytic discrete adjoint remain owned by
//! [`crate::dynamics::stochastic_schrodinger`]. This module adds bounded owned
//! requests, content identities, and fail-closed replay validation for direct
//! Rust callers and language bindings.

use crate::dynamics::stochastic_schrodinger::{
    apply_stochastic_schrodinger_step, audit_stochastic_schrodinger_backward,
    backward_stochastic_schrodinger_step, StochasticSchrodingerBackward,
    StochasticSchrodingerBackwardAudit, StochasticSchrodingerBackwardAuditRequest,
    StochasticSchrodingerConfig, StochasticSchrodingerError, StochasticSchrodingerStep,
    STOCHASTIC_SCHRODINGER_OUTPUT_OBSERVABLE, STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND,
    STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_stochastic_schrodinger_forward.v1";
pub const ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND: &str =
    "spiraltorch.zspace_stochastic_schrodinger_forward";
pub const ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_stochastic_schrodinger_vjp.v1";
pub const ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND: &str =
    "spiraltorch.zspace_stochastic_schrodinger_vjp";
pub const ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER: &str =
    "st-core::runtime::zspace_stochastic_schrodinger";
pub const ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER: &str =
    STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER;
pub const ZSPACE_STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND: &str =
    STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND;
pub const ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE: &str =
    "sha256(contract_version UTF-8 || NUL || compact canonical request JSON)";
pub const ZSPACE_STOCHASTIC_SCHRODINGER_VJP_SEMANTICS: &str =
    "vector-Jacobian product of output_real with respect to input and potential only; standard_normal and config are fixed witnesses; phase is recomputed from the canonical forward request and is never accepted as external evidence";
pub const ZSPACE_STOCHASTIC_SCHRODINGER_EVIDENCE_BOUNDARY: &str =
    "this receipt certifies one bounded real-quadrature numerical transition and its analytic input/potential VJP; it does not establish physical fidelity beyond the stated equation, semantic quality, or training efficacy";

/// Large enough for a four-row GPT-2 vocabulary while keeping browser replay bounded.
pub const ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES: usize = 262_144;
pub const ZSPACE_STOCHASTIC_SCHRODINGER_MAX_ROWS: usize = 262_144;
pub const ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES: usize = 65_536;
pub const ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_BYTES: u64 = 64 * 1_024 * 1_024;
pub const ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_NODES: u64 = 1_500_512;
pub const ZSPACE_STOCHASTIC_SCHRODINGER_MAX_INGRESS_DEPTH: u32 = 10;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceStochasticSchrodingerForwardRequest {
    pub input: Vec<f32>,
    pub potential: Vec<f32>,
    pub standard_normal: Vec<f32>,
    pub rows: usize,
    pub features: usize,
    #[serde(default)]
    pub config: StochasticSchrodingerConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceStochasticSchrodingerForwardReceipt {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub protocol_owner: &'static str,
    pub forward_validated: bool,
    pub forward_id: String,
    pub id_rule: &'static str,
    pub status: &'static str,
    pub request: ZSpaceStochasticSchrodingerForwardRequest,
    pub step: StochasticSchrodingerStep,
    pub efficacy_claim_ready: bool,
    pub evidence_boundary: &'static str,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceStochasticSchrodingerVjpRequest {
    pub forward_request: ZSpaceStochasticSchrodingerForwardRequest,
    pub grad_output_real: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceStochasticSchrodingerVjpReceipt {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub protocol_owner: &'static str,
    pub vjp_validated: bool,
    pub vjp_id: String,
    pub forward_id: String,
    pub id_rule: &'static str,
    pub status: &'static str,
    pub request: ZSpaceStochasticSchrodingerVjpRequest,
    pub output_observable: &'static str,
    pub gradient_semantics: &'static str,
    pub result: StochasticSchrodingerBackward,
    pub audit: StochasticSchrodingerBackwardAudit,
    pub efficacy_claim_ready: bool,
    pub evidence_boundary: &'static str,
}

#[derive(Debug, Error, PartialEq)]
pub enum ZSpaceStochasticSchrodingerProtocolError {
    #[error("stochastic Schrodinger protocol rows must be positive")]
    EmptyRows,
    #[error("stochastic Schrodinger protocol rows {value} exceed maximum {maximum}")]
    RowLimit { value: usize, maximum: usize },
    #[error("stochastic Schrodinger protocol features {value} exceed maximum {maximum}")]
    FeatureLimit { value: usize, maximum: usize },
    #[error("stochastic Schrodinger protocol shape ({rows} x {features}) exceeds usize range")]
    ShapeOverflow { rows: usize, features: usize },
    #[error("stochastic Schrodinger protocol volume {value} exceeds maximum {maximum}")]
    VolumeLimit { value: usize, maximum: usize },
    #[error(
        "stochastic Schrodinger protocol field '{field}' length {value} exceeds maximum {maximum}"
    )]
    FieldLimit {
        field: &'static str,
        value: usize,
        maximum: usize,
    },
    #[error(transparent)]
    Dynamics(#[from] StochasticSchrodingerError),
    #[error("stochastic Schrodinger protocol encoding failed: {message}")]
    Encoding { message: String },
    #[error("malformed stochastic Schrodinger protocol receipt: {message}")]
    MalformedReceipt { message: String },
}

fn malformed(message: impl Into<String>) -> ZSpaceStochasticSchrodingerProtocolError {
    ZSpaceStochasticSchrodingerProtocolError::MalformedReceipt {
        message: message.into(),
    }
}

fn validate_forward_bounds(
    request: &ZSpaceStochasticSchrodingerForwardRequest,
) -> Result<usize, ZSpaceStochasticSchrodingerProtocolError> {
    if request.rows == 0 {
        return Err(ZSpaceStochasticSchrodingerProtocolError::EmptyRows);
    }
    if request.rows > ZSPACE_STOCHASTIC_SCHRODINGER_MAX_ROWS {
        return Err(ZSpaceStochasticSchrodingerProtocolError::RowLimit {
            value: request.rows,
            maximum: ZSPACE_STOCHASTIC_SCHRODINGER_MAX_ROWS,
        });
    }
    if request.features > ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES {
        return Err(ZSpaceStochasticSchrodingerProtocolError::FeatureLimit {
            value: request.features,
            maximum: ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES,
        });
    }
    let volume = request.rows.checked_mul(request.features).ok_or(
        ZSpaceStochasticSchrodingerProtocolError::ShapeOverflow {
            rows: request.rows,
            features: request.features,
        },
    )?;
    if volume > ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES {
        return Err(ZSpaceStochasticSchrodingerProtocolError::VolumeLimit {
            value: volume,
            maximum: ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES,
        });
    }
    for (field, value, maximum) in [
        (
            "input",
            request.input.len(),
            ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES,
        ),
        (
            "standard_normal",
            request.standard_normal.len(),
            ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES,
        ),
        (
            "potential",
            request.potential.len(),
            ZSPACE_STOCHASTIC_SCHRODINGER_MAX_FEATURES,
        ),
    ] {
        if value > maximum {
            return Err(ZSpaceStochasticSchrodingerProtocolError::FieldLimit {
                field,
                value,
                maximum,
            });
        }
    }
    Ok(volume)
}

fn request_id<T: Serialize>(
    contract_version: &'static str,
    request: &T,
) -> Result<String, ZSpaceStochasticSchrodingerProtocolError> {
    let encoded = serde_json::to_vec(request).map_err(|error| {
        ZSpaceStochasticSchrodingerProtocolError::Encoding {
            message: error.to_string(),
        }
    })?;
    let mut hasher = Sha256::new();
    hasher.update(contract_version.as_bytes());
    hasher.update([0]);
    hasher.update(encoded);
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

pub fn run_zspace_stochastic_schrodinger_forward(
    request: ZSpaceStochasticSchrodingerForwardRequest,
) -> Result<ZSpaceStochasticSchrodingerForwardReceipt, ZSpaceStochasticSchrodingerProtocolError> {
    validate_forward_bounds(&request)?;
    let step = apply_stochastic_schrodinger_step(
        &request.input,
        &request.potential,
        &request.standard_normal,
        request.rows,
        request.features,
        request.config,
    )?;
    let forward_id = request_id(
        ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION,
        &request,
    )?;
    Ok(ZSpaceStochasticSchrodingerForwardReceipt {
        contract_version: ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_CONTRACT_VERSION,
        kind: ZSPACE_STOCHASTIC_SCHRODINGER_FORWARD_KIND,
        semantic_owner: STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER,
        semantic_backend: STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND,
        protocol_owner: ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER,
        forward_validated: true,
        forward_id,
        id_rule: ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE,
        status: "ready",
        request,
        step,
        efficacy_claim_ready: false,
        evidence_boundary: ZSPACE_STOCHASTIC_SCHRODINGER_EVIDENCE_BOUNDARY,
    })
}

pub fn validate_zspace_stochastic_schrodinger_forward_value(
    receipt: serde_json::Value,
) -> Result<ZSpaceStochasticSchrodingerForwardReceipt, ZSpaceStochasticSchrodingerProtocolError> {
    let request = receipt
        .get("request")
        .cloned()
        .ok_or_else(|| malformed("missing request"))?;
    let request = serde_json::from_value(request).map_err(|error| malformed(error.to_string()))?;
    let canonical = run_zspace_stochastic_schrodinger_forward(request)?;
    let canonical_value = serde_json::to_value(&canonical).map_err(|error| {
        ZSpaceStochasticSchrodingerProtocolError::Encoding {
            message: error.to_string(),
        }
    })?;
    if !super::canonical_json::values_equivalent(&receipt, &canonical_value) {
        return Err(malformed(
            "receipt does not match the canonical Rust stochastic Schrodinger forward",
        ));
    }
    Ok(canonical)
}

pub fn run_zspace_stochastic_schrodinger_vjp(
    request: ZSpaceStochasticSchrodingerVjpRequest,
) -> Result<ZSpaceStochasticSchrodingerVjpReceipt, ZSpaceStochasticSchrodingerProtocolError> {
    let volume = validate_forward_bounds(&request.forward_request)?;
    if request.grad_output_real.len() > ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES {
        return Err(ZSpaceStochasticSchrodingerProtocolError::FieldLimit {
            field: "grad_output_real",
            value: request.grad_output_real.len(),
            maximum: ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES,
        });
    }
    if request.grad_output_real.len() != volume {
        return Err(StochasticSchrodingerError::LengthMismatch {
            field: "grad_output_real",
            expected: volume,
            actual: request.grad_output_real.len(),
        }
        .into());
    }

    let forward = run_zspace_stochastic_schrodinger_forward(request.forward_request.clone())?;
    let result = backward_stochastic_schrodinger_step(
        &request.forward_request.input,
        &forward.step.phase,
        &request.grad_output_real,
        request.forward_request.rows,
        request.forward_request.features,
        request.forward_request.config,
    )?;
    let audit = audit_stochastic_schrodinger_backward(StochasticSchrodingerBackwardAuditRequest {
        input: &request.forward_request.input,
        phase: &forward.step.phase,
        grad_output: &request.grad_output_real,
        grad_input: &result.grad_input,
        grad_potential: &result.grad_potential,
        rows: request.forward_request.rows,
        features: request.forward_request.features,
        config: request.forward_request.config,
    })?;
    let vjp_id = request_id(ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION, &request)?;
    Ok(ZSpaceStochasticSchrodingerVjpReceipt {
        contract_version: ZSPACE_STOCHASTIC_SCHRODINGER_VJP_CONTRACT_VERSION,
        kind: ZSPACE_STOCHASTIC_SCHRODINGER_VJP_KIND,
        semantic_owner: STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER,
        semantic_backend: STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND,
        protocol_owner: ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER,
        vjp_validated: true,
        vjp_id,
        forward_id: forward.forward_id,
        id_rule: ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE,
        status: "ready",
        request,
        output_observable: STOCHASTIC_SCHRODINGER_OUTPUT_OBSERVABLE,
        gradient_semantics: ZSPACE_STOCHASTIC_SCHRODINGER_VJP_SEMANTICS,
        result,
        audit,
        efficacy_claim_ready: false,
        evidence_boundary: ZSPACE_STOCHASTIC_SCHRODINGER_EVIDENCE_BOUNDARY,
    })
}

pub fn validate_zspace_stochastic_schrodinger_vjp_value(
    receipt: serde_json::Value,
) -> Result<ZSpaceStochasticSchrodingerVjpReceipt, ZSpaceStochasticSchrodingerProtocolError> {
    let request = receipt
        .get("request")
        .cloned()
        .ok_or_else(|| malformed("missing request"))?;
    let request = serde_json::from_value(request).map_err(|error| malformed(error.to_string()))?;
    let canonical = run_zspace_stochastic_schrodinger_vjp(request)?;
    let canonical_value = serde_json::to_value(&canonical).map_err(|error| {
        ZSpaceStochasticSchrodingerProtocolError::Encoding {
            message: error.to_string(),
        }
    })?;
    if !super::canonical_json::values_equivalent(&receipt, &canonical_value) {
        return Err(malformed(
            "receipt does not match the canonical Rust stochastic Schrodinger VJP",
        ));
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> ZSpaceStochasticSchrodingerForwardRequest {
        serde_json::from_value(json!({
            "input": [1.0, 0.25, -0.5, 0.75, 0.1, -0.2],
            "potential": [0.2, -0.1, 0.05],
            "standard_normal": [0.1, -0.3, 0.2, 0.0, 0.4, -0.2],
            "rows": 2,
            "features": 3,
            "config": {
                "time_step": 0.08,
                "hopping_rate": 0.35,
                "loss_rate": 0.02,
                "noise_scale": 0.15
            }
        }))
        .expect("valid request")
    }

    #[test]
    fn forward_is_content_addressed_and_replayable() {
        let receipt = run_zspace_stochastic_schrodinger_forward(request()).expect("forward");
        assert!(receipt.forward_validated);
        assert_eq!(receipt.status, "ready");
        assert_eq!(receipt.step.output_real.len(), 6);
        assert_eq!(receipt.step.output_imaginary.len(), 6);
        assert_eq!(receipt.step.phase.len(), 6);
        assert_eq!(receipt.forward_id.len(), 71);

        let value = serde_json::to_value(&receipt).expect("serializable");
        assert_eq!(
            validate_zspace_stochastic_schrodinger_forward_value(value).expect("replay"),
            receipt
        );
    }

    #[test]
    fn forward_replay_rejects_output_or_request_drift() {
        let receipt = run_zspace_stochastic_schrodinger_forward(request()).expect("forward");
        let mut output_drift = serde_json::to_value(&receipt).expect("serializable");
        output_drift["step"]["output_real"][0] = json!(999.0);
        assert!(matches!(
            validate_zspace_stochastic_schrodinger_forward_value(output_drift),
            Err(ZSpaceStochasticSchrodingerProtocolError::MalformedReceipt { .. })
        ));

        let mut request_drift = serde_json::to_value(&receipt).expect("serializable");
        request_drift["request"]["standard_normal"][0] = json!(0.2);
        assert!(matches!(
            validate_zspace_stochastic_schrodinger_forward_value(request_drift),
            Err(ZSpaceStochasticSchrodingerProtocolError::MalformedReceipt { .. })
        ));
    }

    #[test]
    fn vjp_replays_the_forward_phase_and_rejects_gradient_drift() {
        let request = ZSpaceStochasticSchrodingerVjpRequest {
            forward_request: request(),
            grad_output_real: vec![0.2, -0.4, 0.1, 0.3, 0.0, -0.2],
        };
        let receipt = run_zspace_stochastic_schrodinger_vjp(request).expect("VJP");
        assert!(receipt.vjp_validated);
        assert_eq!(receipt.result.grad_input.len(), 6);
        assert_eq!(receipt.result.grad_potential.len(), 3);
        assert_eq!(receipt.audit.max_grad_input_error, 0.0);
        assert_eq!(receipt.audit.max_grad_potential_error, 0.0);
        assert!(receipt
            .gradient_semantics
            .contains("with respect to input and potential only"));
        assert!(receipt
            .gradient_semantics
            .contains("standard_normal and config are fixed witnesses"));

        let canonical = serde_json::to_value(&receipt).expect("serializable");
        assert_eq!(
            validate_zspace_stochastic_schrodinger_vjp_value(canonical.clone()).expect("replay"),
            receipt
        );
        let mut drifted = canonical;
        drifted["result"]["grad_potential"][0] = json!(2.0);
        assert!(matches!(
            validate_zspace_stochastic_schrodinger_vjp_value(drifted),
            Err(ZSpaceStochasticSchrodingerProtocolError::MalformedReceipt { .. })
        ));
    }

    #[test]
    fn protocol_limits_and_unknown_fields_fail_closed() {
        let oversized = ZSpaceStochasticSchrodingerForwardRequest {
            rows: ZSPACE_STOCHASTIC_SCHRODINGER_MAX_VALUES + 1,
            features: 1,
            input: Vec::new(),
            potential: vec![0.0],
            standard_normal: Vec::new(),
            config: StochasticSchrodingerConfig::default(),
        };
        assert!(matches!(
            run_zspace_stochastic_schrodinger_forward(oversized),
            Err(ZSpaceStochasticSchrodingerProtocolError::RowLimit { .. })
                | Err(ZSpaceStochasticSchrodingerProtocolError::VolumeLimit { .. })
        ));

        assert!(
            serde_json::from_value::<ZSpaceStochasticSchrodingerForwardRequest>(json!({
                "input": [1.0],
                "potential": [0.0],
                "standard_normal": [0.0],
                "rows": 1,
                "features": 1,
                "host_formula": "trust me"
            }))
            .is_err()
        );
    }
}
