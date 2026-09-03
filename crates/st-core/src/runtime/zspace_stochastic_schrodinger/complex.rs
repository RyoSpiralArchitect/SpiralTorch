// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;
use crate::dynamics::stochastic_schrodinger::{
    apply_stochastic_schrodinger_complex_step, backward_stochastic_schrodinger_complex_step,
    StochasticSchrodingerComplexBackward, StochasticSchrodingerComplexInput,
    StochasticSchrodingerComplexStep,
};

pub const ZSPACE_SCHRODINGER_COMPLEX_CONTRACT_VERSION: &str =
    "spiraltorch.zspace_stochastic_schrodinger_complex_step.v1";
pub const ZSPACE_SCHRODINGER_COMPLEX_KIND: &str =
    "spiraltorch.zspace_stochastic_schrodinger_complex_step";
pub const ZSPACE_SCHRODINGER_COMPLEX_GRADIENT_SEMANTICS: &str =
    "real Euclidean VJP of output_real and output_imaginary with respect to input, input_imaginary, and shared potential; config and standard_normal are fixed witnesses";
pub const ZSPACE_SCHRODINGER_COMPLEX_EVIDENCE_BOUNDARY: &str =
    "one complex Strang-split transition and optional analytic input/potential VJP; preserving both quadratures permits composition but does not establish semantic quality or training efficacy";
pub const ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_BYTES: u64 = 96 * 1_024 * 1_024;
pub const ZSPACE_SCHRODINGER_COMPLEX_MAX_INGRESS_NODES: u64 = 3_000_512;

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSchrodingerComplexCotangent {
    pub real: Vec<f32>,
    pub imaginary: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ZSpaceSchrodingerComplexRequest {
    pub forward_request: ZSpaceStochasticSchrodingerForwardRequest,
    pub input_imaginary: Vec<f32>,
    #[serde(default)]
    pub cotangent: Option<ZSpaceSchrodingerComplexCotangent>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ZSpaceSchrodingerComplexReceipt {
    pub contract_version: &'static str,
    pub kind: &'static str,
    pub semantic_owner: &'static str,
    pub semantic_backend: &'static str,
    pub protocol_owner: &'static str,
    pub evaluation_id: String,
    pub id_rule: &'static str,
    pub status: &'static str,
    pub request: ZSpaceSchrodingerComplexRequest,
    pub step: StochasticSchrodingerComplexStep,
    pub gradient: Option<StochasticSchrodingerComplexBackward>,
    pub gradient_semantics: &'static str,
    pub efficacy_claim_ready: bool,
    pub evidence_boundary: &'static str,
}

pub fn run_zspace_stochastic_schrodinger_complex_step(
    request: ZSpaceSchrodingerComplexRequest,
) -> Result<ZSpaceSchrodingerComplexReceipt, ZSpaceStochasticSchrodingerProtocolError> {
    let volume = validate_forward_bounds(&request.forward_request)?;
    let validate_vector =
        |field: &'static str, values: &[f32]| -> Result<(), StochasticSchrodingerError> {
            if values.len() != volume {
                return Err(StochasticSchrodingerError::LengthMismatch {
                    field,
                    expected: volume,
                    actual: values.len(),
                });
            }
            for &value in values {
                if !value.is_finite() {
                    return Err(StochasticSchrodingerError::NonFinite { field, value });
                }
            }
            Ok(())
        };
    validate_vector("input_imaginary", &request.input_imaginary)?;
    if let Some(g) = &request.cotangent {
        validate_vector("grad_output_real", &g.real)?;
        validate_vector("grad_output_imaginary", &g.imaginary)?;
    }
    let forward = &request.forward_request;
    let input = StochasticSchrodingerComplexInput {
        real: &forward.input,
        imaginary: &request.input_imaginary,
        potential: &forward.potential,
        standard_normal: &forward.standard_normal,
        rows: forward.rows,
        features: forward.features,
        config: forward.config,
    };
    let step = apply_stochastic_schrodinger_complex_step(input)?;
    let gradient = request
        .cotangent
        .as_ref()
        .map(|g| backward_stochastic_schrodinger_complex_step(input, &g.real, &g.imaginary))
        .transpose()?;
    Ok(ZSpaceSchrodingerComplexReceipt {
        contract_version: ZSPACE_SCHRODINGER_COMPLEX_CONTRACT_VERSION,
        kind: ZSPACE_SCHRODINGER_COMPLEX_KIND,
        semantic_owner: STOCHASTIC_SCHRODINGER_SEMANTIC_OWNER,
        semantic_backend: STOCHASTIC_SCHRODINGER_SEMANTIC_BACKEND,
        protocol_owner: ZSPACE_STOCHASTIC_SCHRODINGER_PROTOCOL_OWNER,
        evaluation_id: request_id(ZSPACE_SCHRODINGER_COMPLEX_CONTRACT_VERSION, &request)?,
        id_rule: ZSPACE_STOCHASTIC_SCHRODINGER_ID_RULE,
        status: "ready",
        request,
        step,
        gradient,
        gradient_semantics: ZSPACE_SCHRODINGER_COMPLEX_GRADIENT_SEMANTICS,
        efficacy_claim_ready: false,
        evidence_boundary: ZSPACE_SCHRODINGER_COMPLEX_EVIDENCE_BOUNDARY,
    })
}

pub fn validate_zspace_stochastic_schrodinger_complex_value(
    receipt: serde_json::Value,
) -> Result<ZSpaceSchrodingerComplexReceipt, ZSpaceStochasticSchrodingerProtocolError> {
    let request = receipt
        .get("request")
        .cloned()
        .ok_or_else(|| malformed("missing request"))?;
    let request = serde_json::from_value(request).map_err(|error| malformed(error.to_string()))?;
    let canonical = run_zspace_stochastic_schrodinger_complex_step(request)?;
    let expected =
        serde_json::to_value(&canonical).map_err(|error| malformed(error.to_string()))?;
    if !super::super::canonical_json::values_equivalent(&receipt, &expected) {
        return Err(malformed(
            "receipt does not match the canonical Rust complex Schrodinger step",
        ));
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> ZSpaceSchrodingerComplexRequest {
        serde_json::from_value(json!({
            "forward_request": {"input": [0.2, -0.3, 0.7], "potential": [0.1, -0.4, 0.5], "standard_normal": [0.3, 0.7, -0.1], "rows": 1, "features": 3},
            "input_imaginary": [0.6, 0.2, -0.5], "cotangent": {"real": [0.4, -0.1, 0.6], "imaginary": [-0.2, 0.3, 0.8]}
        })).unwrap()
    }

    #[test]
    fn complex_receipt_binds_state_and_both_cotangents() {
        let receipt = run_zspace_stochastic_schrodinger_complex_step(request()).unwrap();
        let value = serde_json::to_value(&receipt).unwrap();
        assert_eq!(
            validate_zspace_stochastic_schrodinger_complex_value(value.clone()).unwrap(),
            receipt
        );
        for pointer in [
            "/request/input_imaginary/0",
            "/request/cotangent/imaginary/0",
            "/step/output_imaginary/0",
            "/gradient/grad_input_imaginary/0",
            "/gradient/grad_potential/0",
        ] {
            let mut altered = value.clone();
            *altered.pointer_mut(pointer).unwrap() = json!(0.99);
            assert!(
                validate_zspace_stochastic_schrodinger_complex_value(altered).is_err(),
                "{pointer}"
            );
        }
        let mut next = request();
        next.input_imaginary[0] += 0.2;
        assert_ne!(
            receipt.evaluation_id,
            run_zspace_stochastic_schrodinger_complex_step(next)
                .unwrap()
                .evaluation_id
        );
        let mut forward_only = request();
        forward_only.cotangent = None;
        assert!(run_zspace_stochastic_schrodinger_complex_step(forward_only)
            .unwrap()
            .gradient
            .is_none());
    }

    #[test]
    fn complex_request_rejects_bad_dimensions_and_unknown_fields() {
        let mut bad = request();
        bad.input_imaginary.clear();
        assert!(run_zspace_stochastic_schrodinger_complex_step(bad).is_err());
        let mut bad = request();
        bad.cotangent.as_mut().unwrap().real[0] = f32::NAN;
        assert!(run_zspace_stochastic_schrodinger_complex_step(bad).is_err());
        let mut bad = request();
        bad.forward_request.rows = ZSPACE_STOCHASTIC_SCHRODINGER_MAX_ROWS + 1;
        assert!(run_zspace_stochastic_schrodinger_complex_step(bad).is_err());
        let mut value = serde_json::to_value(request()).unwrap();
        value["external_phase"] = json!([0.0, 0.0, 0.0]);
        assert!(serde_json::from_value::<ZSpaceSchrodingerComplexRequest>(value).is_err());
    }
}
