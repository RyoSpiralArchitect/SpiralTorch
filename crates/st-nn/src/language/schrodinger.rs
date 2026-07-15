// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::desire::DesireWeights;
use super::geometry::{RepressionField, SemanticBridge, SymbolGeometry};
use crate::{PureResult, TensorError};
use st_core::inference::imaginary_time_schrodinger::{
    apply_imaginary_time_schrodinger, ImaginaryTimeSchrodingerConfig, ImaginaryTimeSchrodingerEdge,
    ImaginaryTimeSchrodingerPayload, ImaginaryTimeSchrodingerRequest,
};
use st_tensor::{emit_tensor_op, emit_tensor_op_meta};
use std::collections::BTreeSet;

fn require_weight(name: &'static str, value: f32) -> PureResult<f64> {
    if !value.is_finite() || value < 0.0 {
        return Err(TensorError::InvalidValue { label: name });
    }
    Ok(f64::from(value))
}

fn hermitian_geometry_edges(
    geometry: &SymbolGeometry,
    alpha: f64,
    beta: f64,
) -> PureResult<(Vec<ImaginaryTimeSchrodingerEdge>, f64)> {
    let dimension = geometry.vocab_size();
    let mut pairs = BTreeSet::new();
    for left in 0..dimension {
        for &(right, log_probability) in geometry.syn_row(left) {
            if right >= dimension {
                return Err(TensorError::InvalidValue {
                    label: "Schrodinger syntax edge index must be in vocabulary",
                });
            }
            if !log_probability.is_finite() {
                return Err(TensorError::InvalidValue {
                    label: "Schrodinger syntax log probability must be finite",
                });
            }
            if left != right {
                pairs.insert((left.min(right), left.max(right)));
            }
        }
    }

    let syntax_exponent = 1.0 + alpha;
    let mut raw_edges = Vec::with_capacity(pairs.len());
    let mut degree = vec![0.0f64; dimension];
    for (left, right) in pairs {
        let directed_log_weight = |source: usize, target: usize| -> PureResult<f64> {
            let log_syntax = f64::from(geometry.log_syn(source, target));
            let log_paradigm = f64::from(geometry.log_par(source, target));
            if !log_syntax.is_finite() || !log_paradigm.is_finite() {
                return Err(TensorError::InvalidValue {
                    label: "Schrodinger geometry log probability must be finite",
                });
            }
            let value = syntax_exponent * log_syntax + beta * log_paradigm;
            if !value.is_finite() {
                return Err(TensorError::InvalidValue {
                    label: "Schrodinger geometry exponent must be finite",
                });
            }
            Ok(value.exp())
        };
        let forward = directed_log_weight(left, right)?;
        let reverse = directed_log_weight(right, left)?;
        let weight = 0.5 * (forward + reverse);
        if !weight.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "Schrodinger Hermitian edge weight must be finite",
            });
        }
        if weight == 0.0 {
            continue;
        }
        degree[left] += weight;
        degree[right] += weight;
        raw_edges.push(ImaginaryTimeSchrodingerEdge {
            left,
            right,
            weight,
        });
    }

    // Preserve graph shape while keeping lookahead time independent of vocabulary degree.
    let affinity_scale = degree.iter().copied().fold(1.0, f64::max);
    for edge in &mut raw_edges {
        edge.weight /= affinity_scale;
    }
    Ok((raw_edges, affinity_scale))
}

fn schrodinger_request(
    geometry: &SymbolGeometry,
    repression: &RepressionField,
    bridge: &SemanticBridge,
    weights: &DesireWeights,
    concept_expectation: &[f32],
    lookahead: usize,
) -> PureResult<(ImaginaryTimeSchrodingerRequest, f64)> {
    let alpha = require_weight(
        "Schrodinger alpha must be finite and non-negative",
        weights.alpha,
    )?;
    let beta = require_weight(
        "Schrodinger beta must be finite and non-negative",
        weights.beta,
    )?;
    let gamma = require_weight(
        "Schrodinger gamma must be finite and non-negative",
        weights.gamma,
    )?;
    let lambda = require_weight(
        "Schrodinger lambda must be finite and non-negative",
        weights.lambda,
    )?;
    let dimension = geometry.vocab_size();
    if repression.len() != dimension || bridge.vocab_size() != dimension {
        return Err(TensorError::InvalidValue {
            label: "Schrodinger fields must align with vocabulary",
        });
    }
    if concept_expectation.len() != bridge.concept_count() {
        return Err(TensorError::InvalidValue {
            label: "Schrodinger concept expectation must match semantic concepts",
        });
    }
    if concept_expectation
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(TensorError::InvalidValue {
            label: "Schrodinger concept expectation must be finite and non-negative",
        });
    }
    let (edges, affinity_scale) = hermitian_geometry_edges(geometry, alpha, beta)?;
    let mut potential = Vec::with_capacity(dimension);
    for token in 0..dimension {
        let repression_value = f64::from(repression.value(token));
        let semantic_expectation = f64::from(bridge.expectation(token, concept_expectation));
        let value = lambda * repression_value - gamma * semantic_expectation;
        if !value.is_finite() {
            return Err(TensorError::InvalidValue {
                label: "Schrodinger scalar potential must be finite",
            });
        }
        potential.push(value);
    }
    let imaginary_time = lookahead as f64;
    if !imaginary_time.is_finite() {
        return Err(TensorError::InvalidValue {
            label: "Schrodinger lookahead must fit finite imaginary time",
        });
    }
    let config = ImaginaryTimeSchrodingerConfig::new(imaginary_time)
        .map_err(|error| TensorError::Generic(error.to_string()))?;
    Ok((
        ImaginaryTimeSchrodingerRequest {
            tags: (0..dimension)
                .map(|token| format!("token:{token}"))
                .collect(),
            potential,
            edges,
            initial_amplitude: Vec::new(),
            config,
        },
        affinity_scale,
    ))
}

fn emit_schrodinger_meta(payload: &ImaginaryTimeSchrodingerPayload, affinity_scale: f64) {
    emit_tensor_op(
        "language_imaginary_time_schrodinger",
        &[payload.tags.len()],
        &[payload.log_amplitude_boost.len()],
    );
    emit_tensor_op_meta("language_imaginary_time_schrodinger", || {
        serde_json::json!({
            "kind": payload.kind,
            "contract_version": payload.contract_version,
            "semantic_owner": payload.semantic_owner,
            "semantic_backend": payload.semantic_backend,
            "backend": payload.backend,
            "execution_backend": payload.execution_backend,
            "route_blocker": payload.route_blocker,
            "vocabulary": payload.tags.len(),
            "edge_count": payload.effects.edge_count,
            "affinity_scale": affinity_scale,
            "imaginary_time": payload.config.imaginary_time(),
            "substeps": payload.effects.substeps,
            "substep_imaginary_time": payload.effects.substep_imaginary_time,
            "spectral_upper_bound": payload.effects.spectral_upper_bound,
            "initial_rayleigh_energy": payload.effects.initial_rayleigh_energy,
            "final_rayleigh_energy": payload.effects.final_rayleigh_energy,
            "rayleigh_energy_drop": payload.effects.rayleigh_energy_drop,
            "energy_tolerance": payload.effects.energy_tolerance,
            "initial_l2_norm": payload.effects.initial_l2_norm,
            "final_l2_norm": payload.effects.final_l2_norm,
            "l2_norm_tolerance": payload.effects.l2_norm_tolerance,
            "initial_residual_l2": payload.effects.initial_residual_l2,
            "final_residual_l2": payload.effects.final_residual_l2,
            "probability_entropy": payload.effects.probability_entropy,
            "probability_sum_tolerance": payload.probability_sum_tolerance,
            "empty": payload.tags.is_empty(),
        })
    });
}

pub(crate) fn schrodinger_boost(
    geometry: &SymbolGeometry,
    repression: &RepressionField,
    bridge: &SemanticBridge,
    weights: &DesireWeights,
    concept_expectation: &[f32],
    lookahead: usize,
) -> PureResult<Vec<f32>> {
    if lookahead == 0 {
        return Ok(vec![0.0; geometry.vocab_size()]);
    }
    let (request, affinity_scale) = schrodinger_request(
        geometry,
        repression,
        bridge,
        weights,
        concept_expectation,
        lookahead,
    )?;
    let payload = apply_imaginary_time_schrodinger(request)
        .map_err(|error| TensorError::Generic(error.to_string()))?;
    emit_schrodinger_meta(&payload, affinity_scale);
    payload
        .log_amplitude_boost
        .iter()
        .map(|&value| {
            let value = value as f32;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(TensorError::InvalidValue {
                    label: "Schrodinger log amplitude must fit f32",
                })
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language::{SemanticBridge, SparseKernel};
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex, OnceLock};

    fn observer_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn geometry() -> SymbolGeometry {
        let syntax = SparseKernel::from_dense(vec![vec![0.8, 0.2], vec![0.4, 0.6]], 1.0e-6)
            .expect("valid syntax");
        let paradigm = SparseKernel::from_dense(vec![vec![0.7, 0.3], vec![0.3, 0.7]], 1.0e-6)
            .expect("valid paradigm");
        SymbolGeometry::new(syntax, paradigm).expect("aligned geometry")
    }

    fn bridge() -> SemanticBridge {
        SemanticBridge::new(
            vec![
                vec![(0, (0.9f32).ln()), (1, (0.1f32).ln())],
                vec![(0, (0.1f32).ln()), (1, (0.9f32).ln())],
            ],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            HashSet::new(),
            1.0e-6,
            SparseKernel::from_dense(vec![vec![1.0, 0.0], vec![0.0, 1.0]], 1.0e-6)
                .expect("valid concept kernel"),
        )
        .expect("valid bridge")
    }

    #[test]
    fn semantic_attraction_becomes_lower_scalar_potential() {
        let geometry = geometry();
        let repression = RepressionField::new(vec![0.0, 0.0]).expect("valid repression");
        let weights = DesireWeights::new(0.0, 0.0, 1.0, 0.0);
        let boosts = schrodinger_boost(&geometry, &repression, &bridge(), &weights, &[1.0, 0.0], 3)
            .expect("valid evolution");

        assert_eq!(boosts[0], 0.0);
        assert!(boosts[1] < 0.0);
    }

    #[test]
    fn repression_becomes_a_repulsive_scalar_potential() {
        let boosts = schrodinger_boost(
            &geometry(),
            &RepressionField::new(vec![0.0, 1.0]).expect("valid repression"),
            &bridge(),
            &DesireWeights::new(0.0, 0.0, 0.0, 1.0),
            &[0.5, 0.5],
            3,
        )
        .expect("valid evolution");

        assert_eq!(boosts[0], 0.0);
        assert!(boosts[1] < 0.0);
    }

    #[test]
    fn zero_lookahead_is_an_exact_zero_boost() {
        let boosts = schrodinger_boost(
            &geometry(),
            &RepressionField::new(vec![0.0, 0.0]).expect("valid repression"),
            &bridge(),
            &DesireWeights::new(1.0, 1.0, 1.0, 1.0),
            &[0.5, 0.5],
            0,
        )
        .expect("disabled evolution");

        assert_eq!(boosts, vec![0.0, 0.0]);
    }

    #[test]
    fn adapter_emits_rust_owned_contract_metadata() {
        let _lock = observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_thread_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));
        let result = schrodinger_boost(
            &geometry(),
            &RepressionField::new(vec![0.1, 0.2]).expect("valid repression"),
            &bridge(),
            &DesireWeights::new(0.2, 0.1, 0.3, 0.05),
            &[0.6, 0.4],
            2,
        );
        st_tensor::set_thread_meta_observer(previous);
        result.expect("valid evolution");

        let events = events.lock().unwrap();
        let event = events
            .iter()
            .find(|(name, _)| *name == "language_imaginary_time_schrodinger")
            .expect("Schrodinger event");
        assert_eq!(
            event.1["semantic_owner"],
            "st-core::inference::imaginary_time_schrodinger"
        );
        assert_eq!(event.1["semantic_backend"], "rust");
        assert_eq!(event.1["execution_backend"], "f64_cpu");
        assert!(
            event.1["final_rayleigh_energy"].as_f64().unwrap()
                <= event.1["initial_rayleigh_energy"].as_f64().unwrap()
        );
    }

    #[test]
    fn adapter_rejects_misaligned_or_non_finite_concept_expectation() {
        let repression = RepressionField::new(vec![0.0, 0.0]).expect("valid repression");
        let weights = DesireWeights::new(0.0, 0.0, 1.0, 0.0);

        let shape_error =
            schrodinger_boost(&geometry(), &repression, &bridge(), &weights, &[1.0], 1)
                .expect_err("concept dimensions must align");
        assert!(shape_error.to_string().contains("semantic concepts"));

        let finite_error = schrodinger_boost(
            &geometry(),
            &repression,
            &bridge(),
            &weights,
            &[f32::NAN, 0.0],
            1,
        )
        .expect_err("concept values must be finite");
        assert!(finite_error.to_string().contains("finite and non-negative"));
    }

    #[test]
    fn adapter_rejects_geometry_edges_outside_the_vocabulary() {
        let syntax = SparseKernel::from_rows(vec![vec![(2, 1.0)], vec![(1, 1.0)]], 1.0e-6)
            .expect("sparse kernel retains external index for adapter validation");
        let paradigm = SparseKernel::from_dense(vec![vec![1.0, 0.0], vec![0.0, 1.0]], 1.0e-6)
            .expect("valid paradigm");
        let malformed = SymbolGeometry::new(syntax, paradigm).expect("row dimensions align");
        let error = schrodinger_boost(
            &malformed,
            &RepressionField::new(vec![0.0, 0.0]).expect("valid repression"),
            &bridge(),
            &DesireWeights::new(0.0, 0.0, 0.0, 0.0),
            &[0.5, 0.5],
            1,
        )
        .expect_err("external geometry edge must fail closed");

        assert!(error
            .to_string()
            .contains("edge index must be in vocabulary"));
    }
}
